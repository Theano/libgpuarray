from mako.template import Template

from tools import ScalarArg, ArrayArg
from dtypes import parse_c_arg_backend

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


# arguments: preamble, name, arguments, expression
generic_kernel = Template("""
${preamble}

KERNEL void ${name}(const unsigned int n,
                    const unsigned int nd,
                    const GLOBAL_MEM unsigned int *dims,
                    const GLOBAL_MEM int *strs
% for arg in arguments:
                    , ${arg.decltype()} ${arg.name}_data
% endfor
) {
  const unsigned int idx = LDIM_0 * GID_0 + LID_0;
  const unsigned int numThreads = LDIM_0 * GDIM_0;
  unsigned int i;

  for (i = idx; i < n; i += numThreads) {
    unsigned int ii = i;
    unsigned int pos;
% for arg in arguments:
    % if arg.isarray():
    GLOBAL_MEM char *${arg.name}_p = (GLOBAL_MEM char *)${arg.name}_data;
    % endif
% endfor
    for (unsigned int d = nd-1; d > 0; d--) {
        pos = ii % dims[d];
        ii = ii / dims[d];
    % for a, arg in enumerate(arguments):
        % if arg.isarray():
        ${arg.name}_p += pos * strs[(nd*${a})+d];
        % endif
    % endfor
    }
    pos = ii;
    % for a, arg in enumerate(arguments):
        % if arg.isarray():
    ${arg.name}_p += pos * strs[nd*${a}];
        % endif
    % endfor
    % for arg in arguments:
    ${arg.decltype()} ${arg.name} = (${arg.decltype()})${arg.name}_p;
    % endfor
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


class ElemwiseKernel(object):
    def __init__(self, kind, context, arguments, operation, name="kernel", preamble="", options=[]):
        if isinstance(arguments, str):
            self.arguments = parse_c_args(arguments)
        else:
            self.arguments = arguments

        if not any(isinstance(arg, ArrayArg) for arg in self.arguments):
            raise RuntimeError(
                "ElemwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")

        extra_preamble = []
        have_bytestore_pragma = False
        have_double_pragma = False
        for arg in self.arguments:
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

            if extra_preamble:
                preamble = '\n'.join(extra_preamble) + '\n' + preamble


import numpy
from ndarray import pygpu_ndarray as gpuarray
kind = "opencl"
ctx = gpuarray.init(kind, 0)
preamble = gpuarray.get_preamble(kind)

def call_elemwise_gen(kind, ctx, args, c_args, operation):
    arguments = parse_c_args(c_args)
    expression = massage_op(operation)
    source = generic_kernel.render(preamble=preamble, name="elemk",
                                   arguments=arguments,
                                   expression=expression)
    k = gpuarray.GpuKernel(source, "elemk", kind=kind, context=ctx)
    kernel_args = []
    n = 0
    nd = 0
    dims = None
    strs = []
    for arg in args:
        if isinstance(arg, gpuarray.GpuArray):
            n = arg.size
            nd = arg.ndim
            dims = arg.shape
            kernel_args.append(arg)
            strs.extend(arg.strides)
#            kernel_args.append(gpuarray.array(arg.strides, dtype='int32',
#                                              order='C',
#                                              kind=kind, context=ctx))
        else:
            kernel_args.append(arg)

    if dims is None:
        raise ValueError("No array arguments")
    if n == 0:
        return # no work to be done.

    kernel_args.insert(0, gpuarray.array(strs, dtype='int32', order='C',
                                         kind=kind, context=ctx))
    kernel_args.insert(0, gpuarray.array(dims, dtype='uint32', order='C',
                                         kind=kind, context=ctx))
    kernel_args.insert(0, numpy.asarray(nd, 'uint32'))
    kernel_args.insert(0, numpy.asarray(n, 'uint32'))
    k(*kernel_args, n=n)

def call_elemwise_basic(kind, ctx, args, c_args, operation):
    arguments = parse_c_args(c_args)
    expression = massage_op(operation)
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
    source = basic_kernel.render(preamble=preamble, name="elemk",
                                 nd=nd, arguments=arguments,
                                 expression=expression)
    k = gpuarray.GpuKernel(source, "elemk", kind=kind, context=ctx)
    k(*kernel_args, n=n)


def call_elemwise_specialized(kind, ctx, args, c_args, operation):
    arguments = parse_c_args(c_args)
    expression = massage_op(operation)
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

    source = specialized_kernel.render(preamble=preamble, name="elemk",
                                       n=n, nd=nd, dim=dims, strs=strs,
                                       arguments=arguments,
                                       expression=expression)
    k = gpuarray.GpuKernel(source, "elemk", kind=kind, context=ctx)
    k(*kernel_args, n=n)

def call_elemwise_contig(kind, ctx, args, c_args, operation):
    arguments = parse_c_args(c_args)
    source = contiguous_kernel.render(preamble=preamble, name="elemk",
                                      arguments=arguments,
                                      expression=operation)
    n = None
    kernel_args = []
    for arg in args:
        if isinstance(arg, gpuarray.GpuArray):
            n = arg.size
            kernel_args.append(arg),
        else:
            kernel_args.append(arg)
    kernel_args.insert(0, numpy.asarray(n, 'uint32'))
    k = gpuarray.GpuKernel(source, "elemk", kind=kind, context=ctx)
    k(*kernel_args, n=n)

a = numpy.random.random((500, 1000)).astype('float32')
b = numpy.random.random((500, 1000)).astype('float32')
c = a + b
gpua = gpuarray.array(a, kind=kind, context=ctx)
gpub = gpuarray.array(b, kind=kind, context=ctx)
gpuc = gpuarray.empty(c.shape, dtype=c.dtype, kind=kind, context=ctx)
def smoketest(f):
    f(kind, ctx, (gpua, gpub, gpuc), "float *a, float *b, float *c", "c[i] = a[i] + b[i]")

smoketest(call_elemwise_gen)
assert (c == numpy.asarray(gpuc)).all()
gpuc = gpuarray.empty(c.shape, dtype=c.dtype, kind=kind, context=ctx)
smoketest(call_elemwise_basic)
assert (c == numpy.asarray(gpuc)).all()
gpuc = gpuarray.empty(c.shape, dtype=c.dtype, kind=kind, context=ctx)
smoketest(call_elemwise_specialized)
assert (c == numpy.asarray(gpuc)).all()
gpuc = gpuarray.empty(c.shape, dtype=c.dtype, kind=kind, context=ctx)
smoketest(call_elemwise_contig)
assert (c == numpy.asarray(gpuc)).all()

import gc, time, math

def timeit(f, lbl):

    gc.disable()
    t = time.time()
    f()
    est = time.time() - t
    gc.enable()

    loops = max(1, int(10**math.floor(math.log(10/est, 10))))

    gc.disable()
    t = time.time()
    for _ in xrange(loops):
        f()

    print lbl, "(", loops, "loops ):", (time.time() - t)/loops, "s"
    gc.enable()

def f_basic():
    smoketest(call_elemwise_basic)

def f_gen():
    smoketest(call_elemwise_gen)

def f_spec():
    smoketest(call_elemwise_specialized)

def f_contig():
    smoketest(call_elemwise_contig)

timeit(f_basic, 'basic')
timeit(f_gen, 'generic')
timeit(f_spec, 'specialized')
timeit(f_contig, 'contiguous')
