from mako.template import Template

from tools import ScalarArg, ArrayArg
from dtypes import parse_c_arg_backend

import numpy
from ndarray import pygpu_ndarray as gpuarray

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
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
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
% endfor
) {
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;

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
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
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
                 spec_limit=3):
        self._cache = dict()

        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        else:
            self.arguments = arguments

        self.operation = operation
        self.expression = massage_op(operation)
        self.kind = kind
        self.context = context
        self._spec_limit = spec_limit

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

    def prepare_args_contig(self, args):
        _, _, _, contig = self.check_args(args)
        if not contig:
            raise RuntimeError("Contig call on not contiguous arrays! halp!")
        self.kernel_args = list(args)
        self.kernel_args.insert(0, numpy.asarray(self.n, 'uint32'))

    def get_basic(self, args, nd, dims):
        self._prepare_args_basic(args, dims)
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

    def prepare_args_basic(cls, args, dims):
        kernel_args = []
        for arg in args:
            if isinstance(arg, gpuarray.GpuArray):
                kernel_args.append(arg),
                kernel_args.extend(numpy.asarray(s, dtype='int32') \
                                       for s in arg.strides)
            else:
                kernel_args.append(arg)

        for d in reversed(dims):
            kernel_args.insert(0, numpy.asarray(d, dtype='uint32'))

        kernel_args.insert(0, numpy.asarray(self.n, dtype='uint32'))

        self.kernel_args = kernel_args
        return nd

    def get_specialized(self, args, nd, dims, str):
        self.prepare_args_specialized(args)
        src = specialized_kernel.render(preamble=self.preamble,
                                        name="elemk", n=self.n, nd=nd,
                                        dim=dims, strs=strs,
                                        arguments=self.arguments,
                                        expression=self.expression)
        return gpuarray.GpuKernel(src, "elemk", kind=self.kind,
                                  context=self.context, cluda=True,
                                  **self.flags)

    def prepare_args_specialized(self, args):
        self.kernel_args = args

    def check_args(self, args):
        arrays = [arg for arg in args if isinstance(arg, gpuarray.GpuArray)]
        if len(arrays) < 1:
            raise ArugmentError("No arrays in kernel arguments, " \
                                    "something is wrong")
        n = arrays[0].size
        nd = arrays[0].ndim
        dims = arrays[0].shape
        strs = [None]*len(args)
        c_contig = True
        f_contig = True
        for arg in arrays[1:]:
            if dims != arg.shape:
                raise ValueError("Some array differs from the others in shape")
            strs.append(arg.strides)
            c_contig = c_contig and arg.flags['C_CONTIGUOUS']
            f_contig = f_contig and arg.flags['F_CONTIGUOUS']

        self.n = n
        return nd, dims, strs, c_contig or f_contig

    def select_kernel(self, args):
        nd, dims, strs, contig = self.check_args(args)
        if contig:
            self.prepare_args_contig(args)
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
                if self._numcall == self._spec_limit:
                    self._speck = self.get_specialized(args, nd, dims, strs)
                    return self._speck
        else:
            self._speckey = key
            self._speck = None
            self._numcall = 1

        if nd not in self._cache:
            k = self.get_basic(args, nd, dims)
            self._cache[nd] = k
            return k
        self.prepare_args_basic(args, dims)
        return self._cache[nd]

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
