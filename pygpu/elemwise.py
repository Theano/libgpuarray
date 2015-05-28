from mako.template import Template

import numpy

from .tools import ScalarArg, ArrayArg, as_argument, check_args, lfu_cache
from .dtypes import (parse_c_arg_backend, dtype_to_ctype, get_np_obj,
                     get_common_dtype)
from . import gpuarray

__all__ = ['ElemwiseKernel', 'elemwise1', 'elemwise2', 'ielemwise2', 'compare']

# parameters: preamble, name, nd, arguments, expression
basic_kernel = Template("""
${preamble}

KERNEL void ${name}(const unsigned int n
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
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}_data; tmp += ${arg.name}_offset;
  ${arg.name}_data = (${arg.decltype()})tmp;
  % endif
% endfor

  for (i = idx; i < n; i += numThreads) {
    int ii = i;
    int pos;
% for arg in arguments:
    % if arg.isarray():
        GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
% for i in range(nd-1, -1, -1):
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
% endfor
    % for arg in arguments:
        % if arg.isarray():
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
        % endif
    % endfor
    ${expression};
  }
}
""")

# parameters: preamble, name, n, nd, dims, arguments, expression
dimspec_kernel = Template("""
${preamble}

KERNEL void ${name}(
% for arg in arguments:
    % if arg.isarray():
                    ${arg.decltype()} ${arg.name}_data,
                    const unsigned int ${arg.name}_offset${'' if nd == 0 and loop.last else ','}
        % for d in range(nd):
                    const int ${arg.name}_str_${d}${'' if (loop.last and loop.parent.last) else ','}
        % endfor
    % else:
                    ${arg.decltype()} ${arg.name}${',' if not loop.last else ''}
    % endif
% endfor
) {
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}_data; tmp += ${arg.name}_offset;
  ${arg.name}_data = (${arg.decltype()})tmp;
  % endif
% endfor

  for (i = idx; i < ${n}; i += numThreads) {
    int ii = i;
    int pos;
% for arg in arguments:
    % if arg.isarray():
        GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
% for i in range(nd-1, -1, -1):
    % if i > 0:
        pos = ii % ${dims[i]};
        ii = ii / ${dims[i]};
    % else:
        pos = ii;
    % endif
    % for arg in arguments:
        % if arg.isarray():
            ${arg.name}_p += pos * ${arg.name}_str_${i};
        % endif
    % endfor
% endfor
    % for arg in arguments:
        % if arg.isarray():
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
        % endif
    % endfor
    ${expression};
  }
}
""")

# arguments: preamble, name, arguments, expression
contiguous_kernel = Template("""
${preamble}

KERNEL void ${name}(const unsigned int n
% for arg in arguments:
                    , ${arg.decltype()} ${arg.name}
  % if arg.isarray():
                    , const unsigned int ${arg.name}_offset
  % endif
% endfor
) {
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for arg in arguments:
  % if arg.isarray():
  tmp = (GLOBAL_MEM char *)${arg.name}; tmp += ${arg.name}_offset;
  ${arg.name} = (${arg.decltype()})tmp;
  % endif
% endfor

  for (i = idx; i < n; i += numThreads) {
    ${expression};
  }
}
""")


# arguments: preamble, name, arguments, n, nd, dim, strs, expression
specialized_kernel = Template("""
${preamble}

KERNEL void ${name}(
% for i, arg in enumerate(arguments):
    % if i != 0:
    ,
    % endif
    % if arg.isarray():
    ${arg.decltype()} ${arg.name}_data
    % else:
    ${arg.decltype()} ${arg.name}
    % endif
% endfor
) {
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;
  GLOBAL_MEM char *tmp;

% for i, arg in enumerate(arguments):
  % if arg.isarray() and offsets[i] != 0:
  tmp = (GLOBAL_MEM char *)${arg.name}_data; tmp += ${offsets[i]};
  ${arg.name}_data = (${arg.decltype()})tmp;
  % endif
% endfor

  for (i = idx; i < ${n}; i += numThreads) {
    int ii = i;
    int pos;
% for arg in arguments:
    % if arg.isarray():
        GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
% for i in range(nd-1, -1, -1):
    % if i > 0:
        pos = ii % ${dim[i]};
        ii = ii / ${dim[i]};
    % else:
        pos = ii;
    % endif
    % for a, arg in enumerate(arguments):
        % if arg.isarray() and strs[a][i] != 0:
            ${arg.name}_p += pos * ${strs[a][i]};
        % endif
    % endfor
% endfor
    % for arg in arguments:
        % if arg.isarray():
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
        % endif
    % endfor
    ${expression};
  }
}
""")


def parse_c_args(arguments):
    return tuple(parse_c_arg_backend(arg, ScalarArg, ArrayArg)
            for arg in arguments.split(','))


import re
INDEX_RE = re.compile('([a-zA-Z_][a-zA-Z0-9_]*)\[i\]')


def massage_op(operation):
    return INDEX_RE.sub('\g<1>[0]', operation)


class ElemwiseKernel(object):
    def __init__(self, context, arguments, operation, preamble="",
                 dimspec_limit=2, spec_limit=10):
        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        else:
            self.arguments = tuple(arguments)

        self.operation = operation
        self.expression = massage_op(operation)
        self.context = context
        self._spec_limit = spec_limit
        self._dimspec_limit = dimspec_limit

        if not any(arg.isarray() for arg in self.arguments):
            raise RuntimeError("ElemwiseKernel can only be used with "
                               "functions that have at least one "
                               "vector argument.")

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

        self.contig_src = contiguous_kernel.render(preamble=self.preamble,
                                                   name="elem_contig",
                                                   arguments=self.arguments,
                                                   expression=self.operation)
        self.contig_k = gpuarray.GpuKernel(self.contig_src, "elem_contig",
                                           self.argspec_contig(),
                                           context=self.context, cluda=True,
                                           **self.flags)
        self._speckey = None
        self._dims = None

    def __hash__(self):
        return (hash(self.arguments) ^ hash(self.operation) ^
                hash(self.context) ^ hash(self.preamble))

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.arguments == other.arguments and
                self.operation == other.operation and
                self.context == other.context and
                self.preamble == other.preamble)

    def clear_caches(self):
        """
        Clears the compiled kernel caches.
        """
        self._make_basic.clear()
        self._make_dimspec.clear()
        self._make_specialized.clear()

    def prepare_args_contig(self, args, n, offsets):
        kernel_args = [n]
        for i, arg in enumerate(args):
            kernel_args.append(arg)
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(offsets[i])
        return kernel_args

    def argspec_contig(self):
        spec = []
        spec.append('uint32')
        for i, arg in enumerate(self.arguments):
            spec.append(arg.spec())
            if arg.isarray():
                spec.append('uint32')
        return spec

    def render_basic(self, nd, name="elemk"):
        return basic_kernel.render(preamble=self.preamble, name=name,
                                   nd=nd, arguments=self.arguments,
                                   expression=self.expression)

    @lfu_cache()
    def _make_basic(self, nd):
        name = "elem_" + str(nd)
        src = self.render_basic(nd, name=name)
        return gpuarray.GpuKernel(src, name, self.argspec_basic(nd),
                                  context=self.context, cluda=True,
                                  **self.flags)

    def prepare_args_basic(self, args, n, dims, strs, offsets):
        kernel_args = [n]
        kernel_args.extend(dims)
        for i, arg in enumerate(args):
            kernel_args.append(arg)
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(offsets[i])
                kernel_args.extend(strs[i])
        return kernel_args

    def argspec_basic(self, nd):
        spec = []
        spec.append('uint32')
        spec.extend('uint32' for _ in range(nd))
        for i, arg in enumerate(self.arguments):
            spec.append(arg.spec())
            if arg.isarray():
                spec.append('uint32')
                spec.extend('int32' for _ in range(nd))
        return spec

    def get_basic(self, args, n, nd, dims, strs, offsets):
        args = self.prepare_args_basic(args, n, dims, strs, offsets)
        return self._make_basic(nd), args

    def try_basic(self, args, n, nd, dims, strs, offsets):
        k = self._make.basic.get(self, nd)
        args = self.prepare_args_basic(args, n, dims, strs, offsets)
        return k, args

    @lfu_cache()
    def _make_dimspec(self, n, nd, dims):
        src = dimspec_kernel.render(preamble=self.preamble, name="elemk",
                                    n=n, nd=nd, dims=dims,
                                    arguments=self.arguments,
                                    expression=self.expression)
        return gpuarray.GpuKernel(src, "elemk", self.argspec_dimspec(nd),
                                  context=self.context, cluda=True,
                                  **self.flags)

    def prepare_args_dimspec(self, args, strs, offsets):
        kernel_args = []
        for i, arg in enumerate(args):
            kernel_args.append(arg),
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(offsets[i])
                kernel_args.extend(strs[i])

        return kernel_args

    def argspec_dimspec(self, nd):
        spec = []
        for i, arg in enumerate(self.arguments):
            spec.append(arg.spec())
            if arg.isarray():
                spec.append('uint32')
                spec.extend('int32' for _ in range(nd))
        return spec

    def get_dimspec(self, args, n, nd, dims, strs, offsets):
        args = self.prepare_args_dimspec(args, strs, offsets)
        return self._make_dimspec(n, nd, dims), args

    def try_dimspec(self, args, n, nd, dims, strs, offsets):
        k = self._make_dimspec.get(self, n, nd, dims)
        args = self.prepare_args_dimspec(args, strs, offsets)
        return k, args

    def prepare_args_specialized(self, args):
        return args

    def argspec_specialized(self):
        return [arg.spec() for arg in self.arguments]

    @lfu_cache()
    def _make_specialized(self, n, nd, dims, strs, offsets):
        src = specialized_kernel.render(preamble=self.preamble,
                                        name="elemk", n=n, nd=nd,
                                        dim=dims, strs=strs,
                                        arguments=self.arguments,
                                        expression=self.expression,
                                        offsets=offsets)
        return gpuarray.GpuKernel(src, "elemk", self.argspec_specialized(),
                                  context=self.context, cluda=True,
                                  **self.flags)

    def get_specialized(self, args, n, nd, dims, strs, offsets):
        args = self.prepare_args_specialized(args)
        return self._make_specialized(n, nd, dims, strs, offsets), args

    def try_specialized(self, args, n, nd, dims, strs, offsets):
        k = self._make_specialized.get(self, n, nd, dims, strs, offsets)
        args = self.prepare_args_specialized(args)
        return k, args

    def select_kernel(self, args, collapse=None, broadcast=False):
        n, nd, dims, strs, offsets, contig = check_args(args,
                                                        collapse=collapse,
                                                        broadcast=broadcast)
        if contig:
            return (self.contig_k, self.prepare_args_contig(args, n, offsets)), n

        try:
            return self.try_specialized(args, n, nd, dims, strs, offsets), n
        except KeyError:
            key = dims, strs, offsets
            if key == self._speckey:
                if self._numcall > self._spec_limit:
                    return self.get_specialized(args, n, nd, dims, strs, offsets), n
                self._numcall += 1
            else:
                self._speckey = key
                self._numcall = 1

        try:
            return self.try_dimspec(args, n, nd, dims, strs, offsets), n
        except KeyError:
            if dims == self._dims:
                if self._dimcall > self._dimspec_limit:
                    return self.get_dimspec(args, n, nd, dims, strs, offsets), n
                self._dimcall += 1
            else:
                self._dims = dims
                self._dimcall = 1

        return self.get_basic(args, n, nd, dims, strs, offsets), n

    def prepare(self, *args, **kwargs):
        n, nd, dims, strs, offsets, contig = check_args(args, **kwargs)
        if contig:
            args = self.prepare_args_contig(args, n, offsets)
            self._prepare_k = self.contig_k
        else:
            args = self.prepare_args_specialized(args)
            self._prepare_k = self.get_specialized(args, n, nd, dims, strs, offsets)

        self._prepare_args = args
        self._prepare_n = n

    def prepared_call(self):
        self._prepare_k.call(self._prepare_n, 0, 0, self._prepare_args)

    def __call__(self, *args, **kwargs):
        (k, args), n = self.select_kernel(args, **kwargs)
        if n != 0:
            k(*args, n=n)

    def call_contig(self, *args):
        n, nd, dims, strs, offsets, contig = check_args(args, collapse=False,
                                                        broadcast=False)
        if not contig:
            raise ValueError("Can't call contig on non-contiguous data")
        if n != 0:
            self.contig_k(*self.prepare_args_contig(args, n, offsets), n=n)

    def call_basic(self, *args, **kwargs):
        n, nd, dims, strs, offsets, _ = check_args(args, **kwargs)
        if n != 0:
            k, args = self.get_basic(args, n, nd, dims, strs, offsets)
            k(*args, n=n)

    def call_dimspec(self, *args, **kwargs):
        n, nd, dims, strs, offsets, _ = check_args(args, **kwargs)
        if n != 0:
            k, args = self.get_dimspec(args, n, nd, dims, strs, offsets)
            k(*args, n=n)

    def call_specialized(self, *args, **kwargs):
        n, nd, dims, strs, offsets, _ = check_args(args, **kwargs)
        if n != 0:
            k, args = self.get_specialized(args, n, nd, dims, strs, offsets)
            k(*args, n=n)


def elemwise1(a, op, oper=None, op_tmpl="res[i] = %(op)sa[i]", out=None):
    a_arg = as_argument(a, 'a')
    args = [ArrayArg(a.dtype, 'res'), a_arg]
    if out is None:
        res = a._empty_like_me()
    else:
        res = out

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = ElemwiseKernel(a.context, args, oper)
    k(res, a)
    return res


def elemwise2(a, op, b, ary, odtype=None, oper=None,
              op_tmpl="res[i] = (%(out_t)s)%(a)s %(op)s (%(out_t)s)%(b)s",
              broadcast=False):
    ndim_extend = True
    if not isinstance(a, gpuarray.GpuArray):
        a = numpy.asarray(a)
        ndim_extend = False
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)
        ndim_extend = False
    if odtype is None:
        odtype = get_common_dtype(a, b, True)

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [ArrayArg(odtype, 'res'), a_arg, b_arg]

    if ndim_extend:
        if a.ndim != b.ndim:
            nd = max(a.ndim, b.ndim)
            if a.ndim < nd:
                a = a.reshape(((1,) * (nd - a.ndim))+a.shape)
            if b.ndim < nd:
                b = b.reshape(((1,) * (nd - b.ndim))+b.shape)
        out_shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        res = gpuarray.empty(out_shape, dtype=odtype, context=ary.context,
                             cls=ary.__class__)
    else:
        res = ary._empty_like_me(dtype=odtype)

    if oper is None:
        oper = op_tmpl % {'a': a_arg.expr(), 'op': op, 'b': b_arg.expr(),
                          'out_t': dtype_to_ctype(odtype)}

    k = ElemwiseKernel(ary.context, args, oper)
    k(res, a, b, broadcast=broadcast)
    return res


def ielemwise2(a, op, b, oper=None, op_tmpl="a[i] = a[i] %(op)s %(b)s",
               broadcast=False):
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [a_arg, b_arg]

    if oper is None:
        oper = op_tmpl % {'op': op, 'b': b_arg.expr()}

    k = ElemwiseKernel(a.context, args, oper)
    k(a, b, broadcast=broadcast)
    return a

def compare(a, op, b, broadcast=False):
    return elemwise2(a, op, b, a, odtype=numpy.dtype('bool'),
                     op_tmpl="res[i] = (%(a)s %(op)s %(b)s)",
                     broadcast=broadcast)
