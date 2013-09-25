from pygpu.gpuarray import GpuArrayException
from pygpu.gpuarray cimport (_GpuArray, GpuArray, GA_NO_ERROR, GpuArray_error,
                             pygpu_copy, GA_ANY_ORDER)

cdef extern from "compyte/buffer_blas.h":
    ctypedef enum cb_transpose:
        cb_no_trans,
        cb_trans,
        cb_conj_trans

cdef extern from "compyte/blas.h":
    int GpuArray_sgemv(cb_transpose trans, float alpha, _GpuArray *A,
                       _GpuArray *X, float beta, _GpuArray *Y)

cdef blas_sgemv(cb_transpose trans, float alpha, GpuArray A, GpuArray X,
                float beta, GpuArray Y):
    cdef int err
    err = GpuArray_sgemv(trans, alpha, &A.ga, &X.ga, beta, &Y.ga);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&A.ga, err), err)

cdef api GpuArray pygpu_blas_sgemv(cb_transpose trans, float alpha, GpuArray A,
                                   GpuArray X, float beta, GpuArray Y):
    blas_sgemv(trans, alpha, A, X, beta, Y)
    return Y

def sgemv(float alpha, GpuArray A, GpuArray X, float beta, GpuArray Y,
          trans=False, overwrite_y=False):
    cdef cb_transpose t
    if not overwrite_y:
        Y = pygpu_copy(Y, GA_ANY_ORDER)
    if trans:
        t = cb_trans
    else:
        t = cb_no_trans
    return pygpu_blas_sgemv(t, alpha, A, X, beta, Y)
