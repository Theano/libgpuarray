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
                    , ${arg.decltype()} ${arg.name}_data
    % if arg.isarray():
        % for d in range(nd):
                        , const int ${arg.name}_str_${d}
        % endfor
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
    ${arg.decltype()} ${arg.name}_data
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

class ElemwiseKernelBase(object):
    def __init__(self, kind, context, arguments, operation, preamble=""):
        self._cache = dict()

        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        else:
            self.arguments = arguments

        self.operation = operation
        self.expression = massage_op(operation)
        self.kind = kind
        self.context = context

        if not any(isinstance(arg, ArrayArg) for arg in self.arguments):
            raise RuntimeError(
                "ElemwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")

        extra_preamble = []
        have_bytestore_pragma = False
        have_double_pragma = False
        for arg in self.arguments:
            # Revise this so that we just pass a parameter and let the
            # backend fuss over the details
            if arg.dtype.itemsize < 4 and type(arg) == ArrayArg \
                    and kind == 'opencl':
                # XXX: On ATI cards we must check that the device
                #      actually supports these options otherwise it
                #      will silently not work
                if not have_bytestore_pragma:
                    extra_preamble.append(
                        "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable")

                    have_bytestore_pragma = True
            if arg.dtype in [numpy.float64, numpy.complex128]:
                if not have_double_pragma:
                    extra_preamble.append("#define COMPYTE_DEFINE_CDOUBLE")
                    if kind == 'opencl':
                        extra_preamble.append(
                            "#pragma OPENCL EXTENSION cl_khr_fp64: enable")
                    have_double_pragma = True

        self.preamble = '\n'.join(extra_preamble) + '\n' + preamble + '\n' \
            + gpuarray.get_preamble(kind)
        self.setup()

    def setup(self):
        pass

    def compile(self, args):
        raise NotImplementedError()

    def __call__(self, *args):
        k = self.compile(args)
        k(*self.kernel_args, n=self.n)

class ElemwiseKernelBasic(ElemwiseKernelBase):
    def compile(self, args):
        nd = self.prepare_args(args)
        if nd not in self._cache:
            src = basic_kernel.render(preamble=self.preamble, name="elemk",
                                      nd=nd, arguments=self.arguments,
                                      expression=self.expression)
            self._cache[nd] = gpuarray.GpuKernel(src, "elemk",
                                                 kind=self.kind,
                                                 context=self.context)
        return self._cache[nd]

    def prepare_args(self, args):
        n = 0
        nd = 0
        dims = None
        kernel_args = []
        for arg in args:
            if isinstance(arg, gpuarray.GpuArray):
                n = arg.size
                nd = arg.ndim
                dims = arg.shape
                kernel_args.append(arg),
                kernel_args.extend(numpy.asarray(s, dtype='int32') \
                                       for s in arg.strides)
            else:
                kernel_args.append(arg)

        for d in reversed(dims):
            kernel_args.insert(0, numpy.asarray(d, dtype='uint32'))

        kernel_args.insert(0, numpy.asarray(n, dtype='uint32'))

        self.n = n
        self.kernel_args = kernel_args
        return nd

class ElemwiseKernelSpecialized(ElemwiseKernelBase):
    def compile(self, args):
        nd, dims, strs = self.prepare_args(args)
        key = nd, dims, tuple(strs)
        if key not in self._cache:
            src = specialized_kernel.render(preamble=self.preamble,
                                            name="elemk", n=self.n, nd=nd,
                                            dim=dims, strs=strs,
                                            arguments=self.arguments,
                                            expression=self.expression)
            self._cache[key] = gpuarray.GpuKernel(src, "elemk",
                                                  kind=self.kind,
                                                  context=self.context)
        return self._cache[key]

    def prepare_args(self, args):
        n = 0
        nd = 0
        dims = None
        kernel_args = []
        strs = []
        for arg in args:
            if isinstance(arg, gpuarray.GpuArray):
                n = arg.size
                nd = arg.ndim
                dims = arg.shape
                kernel_args.append(arg)
                strs.append(arg.strides)
            else:
                kernel_args.append(arg)
                strs.append(None)
        self.n = n
        self.kernel_args = kernel_args
        return nd, dims, strs

class ElemwiseKernelContig(ElemwiseKernelBase):
    def setup(self):
        src = contiguous_kernel.render(preamble=self.preamble, name="elemk",
                                       arguments=self.arguments,
                                       expression=self.operation)
        self.k = gpuarray.GpuKernel(src, "elemk", kind=self.kind,
                                    context=self.context)

    def compile(self, args):
        n = None
        kernel_args = []
        for arg in args:
            if isinstance(arg, gpuarray.GpuArray):
                n = arg.size
                kernel_args.append(arg),
            else:
                kernel_args.append(arg)

        kernel_args.insert(0, numpy.asarray(n, dtype='uint32'))

        self.kernel_args = kernel_args
        self.n = n

        return self.k
