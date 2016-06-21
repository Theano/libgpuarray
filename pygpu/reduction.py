import math
import re

from mako.template import Template

import numpy

from . import gpuarray
from .tools import ScalarArg, ArrayArg, check_args, prod, lru_cache
from .dtypes import parse_c_arg_backend


def parse_c_args(arguments):
    return tuple(parse_c_arg_backend(arg, ScalarArg, ArrayArg)
                 for arg in arguments.split(','))


INDEX_RE = re.compile('([a-zA-Z_][a-zA-Z0-9_]*)\[i\]')


def massage_op(operation):
    return INDEX_RE.sub('\g<1>[0]', operation)


def _ceil_log2(x):
    # nearest power of 2 (going up)
    if x != 0:
        return int(math.ceil(math.log(x, 2)))
    else:
        return 0

basic_kernel = Template("""
${preamble}

#define REDUCE(a, b) (${reduce_expr})

KERNEL void ${name}(const unsigned int n, ${out_arg.decltype()} out
% for d in range(nd):
                    , const unsigned int dim${d}
% endfor
% for arg in arguments:
    % if arg.isarray():
                    , ${arg.decltype()} ${arg.name}_data
                    , const unsigned int ${arg.name}_offset
        % for d in range(nd):
                    , const int ${arg.name}_str_${d}
        % endfor
    % else:
                    , ${arg.decltype()} ${arg.name}
    % endif
% endfor
) {
  LOCAL_MEM ${out_arg.ctype()} ldata[${local_size}];
  const unsigned int lid = LID_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}_data; tmp += ${arg.name}_offset;
  ${arg.name}_data = (${arg.decltype()})tmp;
  % endif
% endfor

  i = GID_0;
% for i in range(nd-1, -1, -1):
  % if not redux[i]:
    % if i > 0:
  const unsigned int pos${i} = i % dim${i};
  i = i / dim${i};
    % else:
  const unsigned int pos${i} = i;
    % endif
  % endif
% endfor

  ${out_arg.ctype()} acc = ${neutral};

  for (i = lid; i < n; i += LDIM_0) {
    int ii = i;
    int pos;
% for arg in arguments:
    % if arg.isarray():
        GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
% for i in range(nd-1, -1, -1):
    % if redux[i]:
        % if i > 0:
        pos = ii % dim${i};
        ii = ii / dim${i};
        % else:
        pos = ii;
        % endif
        % for arg in arguments:
            % if arg.isarray():
        ${arg.name}_p += pos * ${arg.name}_str_${i};
            % endif
        % endfor
    % else:
        % for arg in arguments:
            % if arg.isarray():
        ${arg.name}_p += pos${i} * ${arg.name}_str_${i};
            % endif
        % endfor
    % endif
% endfor
% for arg in arguments:
    % if arg.isarray():
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
    % endif
% endfor
    acc = REDUCE((acc), (${map_expr}));
  }
  ldata[lid] = acc;

  <% cur_size = local_size %>
  % while cur_size > 1:
    <% cur_size = cur_size // 2 %>
    local_barrier();
    if (lid < ${cur_size}) {
      ldata[lid] = REDUCE(ldata[lid], ldata[lid+${cur_size}]);
    }
  % endwhile
  if (lid == 0) out[GID_0] = ldata[0];
}
""")


class ReductionKernel(object):
    def __init__(self, context, dtype_out, neutral, reduce_expr, redux,
                 map_expr=None, arguments=None, preamble="", init_nd=None):
        """
        :param init_nd: used to pre compile the reduction code for
            this value of nd and the self.init_local_size value.

        """
        self.context = context
        self.neutral = neutral
        self.redux = tuple(redux)
        if not any(self.redux):
            raise ValueError("Reduction is along no axes")
        self.dtype_out = dtype_out
        self.out_arg = ArrayArg(numpy.dtype(self.dtype_out), 'out')

        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        elif arguments is None:
            self.arguments = [ArrayArg(numpy.dtype(self.dtype_out), '_reduce_input')]
        else:
            self.arguments = arguments

        self.reduce_expr = reduce_expr
        if map_expr is None:
            if len(self.arguments) != 1:
                raise ValueError("Don't know what to do with more than one "
                                 "argument. Specify map_expr to explicitly "
                                 "state what you want.")
            self.operation = "%s[i]" % (self.arguments[0].name,)
            self.expression = "%s[0]" % (self.arguments[0].name,)
        else:
            self.operation = map_expr
            self.expression = massage_op(map_expr)

        if not any(isinstance(arg, ArrayArg) for arg in self.arguments):
            raise ValueError("ReductionKernel can only be used with "
                             "functions that have at least one vector "
                             "argument.")

        have_small = False
        have_double = False
        have_complex = False
        for arg in self.arguments:
            if arg.dtype.itemsize < 4 and type(arg) == ArrayArg:
                have_small = True
            if arg.dtype in [numpy.float64, numpy.complex128]:
                have_double = True
            if arg.dtype in [numpy.complex64, numpy.complex128]:
                have_complex = True

        self.flags = dict(have_small=have_small, have_double=have_double,
                          have_complex=have_complex)
        self.preamble = preamble

        self.init_local_size = min(context.lmemsize //
                                   self.out_arg.dtype.itemsize,
                                   context.maxlsize)

        # this is to prep the cache
        if init_nd is not None:
            self._get_basic_kernel(self.init_local_size, init_nd)

    def _find_kernel_ls(self, tmpl, max_ls, *tmpl_args):
        local_size = min(self.init_local_size, max_ls)
        count_lim = _ceil_log2(local_size)
        local_size = int(2**count_lim)
        loop_count = 0
        while loop_count <= count_lim:
            k, src, spec = tmpl(local_size, *tmpl_args)

            if local_size <= k.maxlsize:
                return k, src, spec, local_size
            else:
                local_size //= 2

            loop_count += 1

        raise RuntimeError("Can't stabilize the local_size for kernel."
                           " Please report this along with your "
                           "reduction code.")

    def _gen_basic(self, ls, nd):
        src = basic_kernel.render(preamble=self.preamble,
                                  reduce_expr=self.reduce_expr,
                                  name="reduk",
                                  out_arg=self.out_arg,
                                  nd=nd, arguments=self.arguments,
                                  local_size=ls,
                                  redux=self.redux,
                                  neutral=self.neutral,
                                  map_expr=self.expression)
        spec = ['uint32', gpuarray.GpuArray]
        spec.extend('uint32' for _ in range(nd))
        for i, arg in enumerate(self.arguments):
            spec.append(arg.spec())
            if arg.isarray():
                spec.append('uint32')
                spec.extend('int32' for _ in range(nd))
        k = gpuarray.GpuKernel(src, "reduk", spec, context=self.context,
                               cluda=True, **self.flags)
        return k, src, spec

    @lru_cache()
    def _get_basic_kernel(self, maxls, nd):
        return self._find_kernel_ls(self._gen_basic, maxls, nd)

    def __call__(self, *args, **kwargs):
        broadcast = kwargs.pop('broadcast', None)
        out = kwargs.pop('out', None)
        if len(kwargs) != 0:
            raise TypeError('Unexpected keyword argument: %s' %
                            kwargs.keys()[0])

        _, nd, dims, strs, offsets = check_args(args, collapse=False,
                                                broadcast=broadcast)

        n = prod(dims)
        out_shape = tuple(d for i, d in enumerate(dims) if not self.redux[i])
        gs = prod(out_shape)
        if gs == 0:
            gs = 1
        n /= gs
        if gs > self.context.maxgsize:
            raise ValueError("Array too big to be reduced along the "
                             "selected axes")

        if out is None:
            out = gpuarray.empty(out_shape, context=self.context,
                                 dtype=self.dtype_out)
        else:
            if out.shape != out_shape or out.dtype != self.dtype_out:
                raise TypeError(
                    "Out array is not of expected type (expected %s %s, "
                    "got %s %s)" % (out_shape, self.dtype_out, out.shape,
                                    out.dtype))
        # Don't compile and cache for nothing for big size
        if self.init_local_size < n:
            k, _, _, ls = self._get_basic_kernel(self.init_local_size, nd)
        else:
            k, _, _, ls = self._get_basic_kernel(2**_ceil_log2(n), nd)

        kargs = [n, out]
        kargs.extend(dims)
        for i, arg in enumerate(args):
            kargs.append(arg)
            if isinstance(arg, gpuarray.GpuArray):
                kargs.append(offsets[i])
                kargs.extend(strs[i])

        k(*kargs, ls=ls, gs=gs)

        return out


def reduce1(ary, op, neutral, out_type, axis=None, out=None, oper=None):
    nd = ary.ndim
    if axis is None:
        redux = [True] * nd
    else:
        redux = [False] * nd

        if not isinstance(axis, (list, tuple)):
            axis = (axis,)

        for ax in axis:
            if ax < 0:
                ax += nd
            if ax < 0 or ax >= nd:
                raise ValueError('axis out of bounds')
            redux[ax] = True

    if oper is None:
        reduce_expr = "a %s b" % (op,)
    else:
        reduce_expr = oper

    r = ReductionKernel(ary.context, dtype_out=out_type, neutral=neutral,
                        reduce_expr=reduce_expr, redux=redux,
                        arguments=[ArrayArg(ary.dtype, 'a')])
    return r(ary, out=out)
