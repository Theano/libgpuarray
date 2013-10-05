from pygpu.gpuarray import GpuArrayException
from pygpu.gpuarray cimport (_GpuArray, GpuArray, GA_NO_ERROR, GpuArray_error,
                             pygpu_copy, pygpu_empty, GA_ANY_ORDER, GA_F_ORDER,
                             GpuArray_ISONESEGMENT)

cdef extern from "compyte/buffer_blas.h":
    ctypedef enum cb_transpose:
        cb_no_trans,
        cb_trans,
        cb_conj_trans

cdef extern from "compyte/blas.h":
    int GpuArray_rgemv(cb_transpose trans, double alpha, const _GpuArray *A,
                       const _GpuArray *X, double beta, _GpuArray *Y,
                       int nocopy)

cdef blas_rgemv(cb_transpose trans, double alpha, GpuArray A, GpuArray X,
                double beta, GpuArray Y, bint nocopy):
    cdef int err
    err = GpuArray_rgemv(trans, alpha, &A.ga, &X.ga, beta, &Y.ga, nocopy);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&A.ga, err), err)

cdef api GpuArray pygpu_blas_rgemv(cb_transpose trans, double alpha,
                                   GpuArray A, GpuArray X, double beta,
                                   GpuArray Y):
    blas_rgemv(trans, alpha, A, X, beta, Y, 0)
    return Y

def gemv(double alpha, GpuArray A, GpuArray X, double beta=0.0,
         GpuArray Y=None, trans=False, overwrite_y=False):
    cdef cb_transpose t
    cdef size_t Yshp
    if A.ga.nd != 2:
        raise TypeError, "A is not a matrix"
    if trans:
        t = cb_trans
        Yshp = A.ga.dimensions[1]
    else:
        t = cb_no_trans
        Yshp = A.ga.dimensions[0]
    if Y is None:
        if beta != 0.0:
            raise ValueError, "Y not provided and beta != 0"
        Y = pygpu_empty(1, &Yshp, A.ga.typecode, GA_ANY_ORDER, A.context, None)
        overwrite_y = True
    if not overwrite_y:
        Y = pygpu_copy(Y, GA_ANY_ORDER)
    return pygpu_blas_rgemv(t, alpha, A, X, beta, Y)
