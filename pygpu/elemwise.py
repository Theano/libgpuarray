from mako.template import Template

from tools import ScalarArg, ArrayArg, as_argument
from dtypes import parse_c_arg_backend

import numpy
import gpuarray
from dtypes import dtype_to_ctype, get_np_obj, get_common_dtype

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
  % if arg.isarray():
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
        % if arg.isarray():
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
    return [parse_c_arg_backend(arg, ScalarArg, ArrayArg)
            for arg in arguments.split(',')]


import re
INDEX_RE = re.compile('([a-zA-Z_][a-zA-Z0-9_]*)\[i\]')


def massage_op(operation):
    return INDEX_RE.sub('\g<1>[0]', operation)


class ElemwiseKernel(object):
    def __init__(self, kind, context, arguments, operation, preamble="",
                 dimspec_limit=2, spec_limit=10):
        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        else:
            self.arguments = arguments

        self.operation = operation
        self.expression = massage_op(operation)
        self.kind = kind
        self.context = context
        self._spec_limit = spec_limit
        self._dimspec_limit = dimspec_limit

        if not any(isinstance(arg, ArrayArg) for arg in self.arguments):
            raise RuntimeError(
                "ElemwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")

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

        src = contiguous_kernel.render(preamble=self.preamble,
                                       name="elemk",
                                       arguments=self.arguments,
                                       expression=self.operation)
        self.contig_k = gpuarray.GpuKernel(src, "elemk", kind=self.kind,
                                           context=self.context, cluda=True,
                                           **self.flags)
        self._speckey = None
        self._dims = None
        self._cache_basic = {}
        self._cache_dimspec = {}

    def prepare_args_contig(self, args, offsets):
        kernel_args = []
        for i, arg in enumerate(args):
            kernel_args.append(arg)
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(numpy.asarray(offsets[i], dtype='uint32'))
        self.kernel_args = kernel_args
        self.kernel_args.insert(0, numpy.asarray(self.n, dtype='uint32'))

    def get_basic(self, args, nd, dims, strs, offsets):
        self.prepare_args_basic(args, dims, strs, offsets)
        if nd not in self._cache_basic:
            src = basic_kernel.render(preamble=self.preamble, name="elemk",
                                      nd=nd, arguments=self.arguments,
                                      expression=self.expression)
            self._cache_basic[nd] = gpuarray.GpuKernel(src, "elemk",
                                                       cluda=True,
                                                       kind=self.kind,
                                                       context=self.context,
                                                       **self.flags)
        return self._cache_basic[nd]

    def prepare_args_basic(self, args, dims, strs, offsets):
        kernel_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(arg),
                kernel_args.append(numpy.asarray(offsets[i], dtype='uint32'))
                kernel_args.extend(numpy.asarray(s, dtype='int32')
                                   for s in strs[i])
            else:
                kernel_args.append(arg)

        for d in reversed(dims):
            kernel_args.insert(0, numpy.asarray(d, dtype='uint32'))

        kernel_args.insert(0, numpy.asarray(self.n, dtype='uint32'))

        self.kernel_args = kernel_args

    def get_dimspec(self, args, nd, dims, strs, offsets):
        self.prepare_args_dimspec(args, strs, offsets)
        if dims not in self._cache_dimspec:
            src = dimspec_kernel.render(preamble=self.preamble, name="elemk",
                                        n=self.n, nd=nd, dims=dims,
                                        arguments=self.arguments,
                                        expression=self.expression)
            self._cache_dimspec[dims] = gpuarray.GpuKernel(src, "elemk",
                                                           cluda=True,
                                                           kind=self.kind,
                                                           context=self.context,
                                                           **self.flags)
        return self._cache_dimspec[dims]

    def prepare_args_dimspec(self, args, strs, offsets):
        kernel_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(arg),
                kernel_args.append(numpy.asarray(offsets[i], dtype='uint32'))
                kernel_args.extend(numpy.asarray(s, dtype='int32')
                                   for s in strs[i])
            else:
                kernel_args.append(arg)

        self.kernel_args = kernel_args

    def get_specialized(self, args, nd, dims, strs, offsets):
        self.prepare_args_specialized(args)
        src = specialized_kernel.render(preamble=self.preamble,
                                        name="elemk", n=self.n, nd=nd,
                                        dim=dims, strs=strs,
                                        arguments=self.arguments,
                                        expression=self.expression,
                                        offsets=offsets)
        return gpuarray.GpuKernel(src, "elemk", kind=self.kind,
                                  context=self.context, cluda=True,
                                  **self.flags)

    def prepare_args_specialized(self, args):
        self.kernel_args = args

    def check_args(self, args, collapse=True):
        arrays = []
        strs = []
        offsets = []
        for arg in args:
            if isinstance(arg, gpuarray.GpuArray):
                strs.append(arg.strides)
                offsets.append(arg.offset)
                arrays.append(arg)
            else:
                strs.append(None)
                offsets.append(None)

        if len(arrays) < 1:
            raise ArugmentError("No arrays in kernel arguments, "
                                "something is wrong")
        n = arrays[0].size
        nd = arrays[0].ndim
        dims = arrays[0].shape
        c_contig = True
        f_contig = True
        for ary in arrays:
            if dims != ary.shape:
                raise ValueError("Some array differs from the others in shape")
            c_contig = c_contig and ary.flags['C_CONTIGUOUS']
            f_contig = f_contig and ary.flags['F_CONTIGUOUS']

        contig = c_contig or f_contig

        if not contig and collapse and nd > 1:
            # make the strides and dims editable
            dims = list(dims)
            strs = [list(str) for str in strs]
            # remove dimensions that are of size 1
            for i in range(nd-1, -1, -1):
                if dims[i] == 1:
                    del dims[i]
                    for str in strs:
                        del str[i]
                    nd -= 1

            # collapse contiguous dimensions
            for i in range(nd-1, 0, -1):
                if all(str[i] * dims[i] == str[i-1] for str in strs):
                    dims[i-1] *= dims[i]
                    del dims[i]
                    for str in strs:
                        str[i-1] = str[i]
                        del str[i]
                    nd -= 1

        self.n = n
        return nd, dims, strs, offsets, contig

    def select_kernel(self, args):
        nd, dims, strs, offsets, contig = self.check_args(args)
        if contig:
            self.prepare_args_contig(args, offsets)
            return self.contig_k

        # If you call self._spec_limit times (default: 3) in a row
        # with the same (or compatible) arguments then we will compile
        # a specialized kernel and use it otherwise we just note the
        # fact and use the basic one.
        key = nd, dims, strs
        if key == self._speckey:
            if self._speck:
                self.prepare_args_specialized(args)
                return self._speck
            else:
                self._numcall += 1
                if self._numcall > self._spec_limit:
                    self._speck = self.get_specialized(args, nd, dims, strs,
                                                       offsets)
                    return self._speck
        elif dims in self._cache_dimspec:
            self.prepare_args_dimspec(args, strs, offsets)
            return self._cache_dimspec[dims]
        elif dims == self._dims:
            self._dimcall += 1
            if self._dimcall > self._dimspec_limit:
                return self.get_dimspec(args, nd, dims, strs, offsets)
        else:
            self._dims = dims
            self._dimcall = 1
            self._speckey = key
            self._speck = None
            self._numcall = 1

        return self.get_basic(args, nd, dims, strs, offsets)

    def prepare(self, *args):
        nd, dims, strs, contig = self.check_args(args)
        if contig:
            self.prepare_args_contig(args)
            self._prepare_k = self.contig_k
        else:
            self.prepare_args_specialized(args)
            self._prepare_k = self.get_specialized(args, nd, dims, strs)

        self._prepare_k.setargs(self.kernel_args)

    def prepared_call(self):
        self._prepare_k.call(self.n)

    def __call__(self, *args):
        k = self.select_kernel(args)
        k(*self.kernel_args, n=self.n)

    def call_contig(self, *args):
        nd, dims, strs, offsets, contig = self.check_args(args)
        if not contig:
            raise ValueError("Can't call contig on non-contiguous data")
        self.prepare_args_contig(args, offsets)
        self.contig_k(*self.kernel_args, n=self.n)

    def call_basic(self, *args):
        nd, dims, strs, offsets, _ = self.check_args(args)
        k = self.get_basic(args, nd, dims, strs, offsets)
        k(*self.kernel_args, n=self.n)

    def call_dimspec(self, *args):
        nd, dims, strs, offsets, _ = self.check_args(args)
        k = self.get_dimspec(args, nd, dims, strs, offsets)
        k(*self.kernel_args, n=self.n)

    def call_specialized(self, *args):
        nd, dims, strs, offsets, _ = self.check_args(args)
        k = self.get_specialized(args, nd, dims, strs, offsets)
        k(*self.kernel_args, n=self.n)


def elemwise1(a, op, oper=None, op_tmpl="res[i] = %(op)sa[i]"):
    a_arg = as_argument(a, 'a')
    args = [ArrayArg(a.dtype, 'res'), a_arg]
    res = a._empty_like_me()

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = ElemwiseKernel(a.kind, a.context, args, oper)
    k(res, a)
    return res


def elemwise2(a, op, b, ary, odtype=None, oper=None,
              op_tmpl="res[i] = (%(out_t)s)%(a)s %(op)s (%(out_t)s)%(b)s"):
    if not isinstance(a, gpuarray.GpuArray):
        a = numpy.asarray(a)
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)
    if odtype is None:
        odtype = get_common_dtype(a, b, True)

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [ArrayArg(odtype, 'res'), a_arg, b_arg]
    res = ary._empty_like_me(dtype=odtype)

    if oper is None:
        oper = op_tmpl % {'a': a_arg.expr(), 'op': op, 'b': b_arg.expr(),
                          'out_t': dtype_to_ctype(odtype)}

    k = ElemwiseKernel(ary.kind, ary.context, args, oper)
    k(res, a, b)
    return res


def ielemwise2(a, op, b, oper=None, op_tmpl="a[i] = a[i] %(op)s %(b)s"):
    if not isinstance(b, gpuarray.GpuArray):
        b = numpy.asarray(b)

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [a_arg, b_arg]

    if oper is None:
        oper = op_tmpl % {'op': op, 'b': b_arg.expr()}

    k = ElemwiseKernel(a.kind, a.context, args, oper)
    k(a, b)
    return a

def compare(a, op, b):
    return elemwise2(a, op, b, a, odtype=numpy.dtype('bool'),
                     op_tmpl="res[i] = (%(a)s %(op)s %(b)s)")
