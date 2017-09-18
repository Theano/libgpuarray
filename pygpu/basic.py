from string import Template
from .gpuarray import GpuArray, GpuKernel, SIZE


def _generate_kernel(ctx, cols, upper=True):
    tmpl = Template("""
    #include "cluda.h"
    KERNEL void extract_tri(GLOBAL_MEM ga_float *a, ga_size a_off, ga_uint N) {
        a = (GLOBAL_MEM ga_float *)(((GLOBAL_MEM char *)a) + a_off);
        unsigned int idx = GID_1 * LDIM_0 * GDIM_0 +
                           GID_0 * LDIM_0 + LID_0;
        unsigned int ix = idx/${cols};
        unsigned int iy = idx%${cols};
        if (idx < N) {
            if (ix ${le} iy)
                a[idx] = 0.0;
        }
    }
    """)
    if upper:
        le = '>'
    else:
        le = '<'
    src = tmpl.substitute(cols=cols, le=le)
    spec = [GpuArray, SIZE, 'uint32']
    k = GpuKernel(src, "extract_tri", spec, context=ctx)
    return k


def triu(A, inplace=True):
    if A.ndim != 2:
        raise ValueError("triu only works for 2d arrays")
    if A.flags.c_contiguous is A.flags.f_contiguous is False:
        raise ValueError("triu only works for contiguous arrays")

    if not inplace:
        A = A.copy()
    if A.flags['F_CONTIGUOUS']:
        upper = False
        cols = A.shape[0]
    else:
        upper = True
        cols = A.shape[1]
    k = _generate_kernel(A.context, cols, upper)
    k(A, A.offset, A.shape[0] * A.shape[1], n=A.shape[0] * A.shape[1])
    return A


def tril(A, inplace=True):
    if A.ndim != 2:
        raise ValueError("tril only works for 2d arrays")
    if A.flags.c_contiguous is A.flags.f_contiguous is False:
        raise ValueError("tril only works for contiguous arrays")

    if not inplace:
        A = A.copy()
    if A.flags['F_CONTIGUOUS']:
        upper = True
        cols = A.shape[0]
    else:
        upper = False
        cols = A.shape[1]
    k = _generate_kernel(A.context, cols, upper)
    k(A, A.offset, A.shape[0] * A.shape[1], n=A.shape[0] * A.shape[1])
    return A
