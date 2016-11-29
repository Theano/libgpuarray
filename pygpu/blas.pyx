from pygpu.gpuarray import GpuArrayException
from pygpu.gpuarray cimport (_GpuArray, GpuArray, GA_NO_ERROR, GpuArray_error,
                             pygpu_copy, pygpu_empty, pygpu_zeros,
                             GA_ANY_ORDER, GA_F_ORDER, GpuArray_ISONESEGMENT)

cdef extern from "gpuarray/buffer_blas.h":
    ctypedef enum cb_transpose:
        cb_no_trans,
        cb_trans,
        cb_conj_trans

cdef extern from "gpuarray/blas.h":
    int GpuArray_rdot(_GpuArray *X, _GpuArray *Y, _GpuArray *Z, int nocopy)
    int GpuArray_rgemv(cb_transpose transA, double alpha, _GpuArray *A,
                       _GpuArray *X, double beta, _GpuArray *Y, int nocopy)
    int GpuArray_rgemm(cb_transpose transA, cb_transpose transB,
                       double alpha, _GpuArray *A, _GpuArray *B,
                       double beta, _GpuArray *C, int nocopy)
    int GpuArray_rger(double alpha, _GpuArray *X, _GpuArray *Y, _GpuArray *A,
                      int nocopy)

cdef api int pygpu_blas_rdot(GpuArray X, GpuArray Y, GpuArray Z, bint nocopy) except -1:
    cdef int err
    err = GpuArray_rdot(&X.ga, &Y.ga, &Z.ga, nocopy)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&X.ga, err), err)
    return 0

cdef api int pygpu_blas_rgemv(cb_transpose transA, double alpha, GpuArray A,
                              GpuArray X, double beta, GpuArray Y,
                              bint nocopy) except -1:
    cdef int err
    err = GpuArray_rgemv(transA, alpha, &A.ga, &X.ga, beta, &Y.ga, nocopy);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&A.ga, err), err)
    return 0

cdef api int pygpu_blas_rgemm(cb_transpose transA, cb_transpose transB,
                              double alpha, GpuArray A, GpuArray B,
                              double beta, GpuArray C, bint nocopy) except -1:
    cdef int err
    err = GpuArray_rgemm(transA, transB, alpha, &A.ga, &B.ga, beta, &C.ga, nocopy);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&A.ga, err), err)
    return 0

cdef api int pygpu_blas_rger(double alpha, GpuArray X, GpuArray Y, GpuArray A,
                             bint nocopy) except -1:
    cdef int err
    err = GpuArray_rger(alpha, &X.ga, &Y.ga, &A.ga, nocopy);
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&X.ga, err), err)
    return 0


def dot(GpuArray X, GpuArray Y, GpuArray Z=None, overwrite_z=False):
    if Z is None:
        Z = pygpu_empty(0, NULL, X.typecode, GA_ANY_ORDER, X.context, None)
        overwrite_z = True

    if not overwrite_z:
        Z = pygpu_copy(Z, GA_ANY_ORDER)
    pygpu_blas_rdot(X, Y, Z, 0)
    return Z

def gemv(double alpha, GpuArray A, GpuArray X, double beta=0.0,
         GpuArray Y=None, trans_a=False, overwrite_y=False):
    cdef cb_transpose transA
    cdef size_t Yshp

    if trans_a:
        transA = cb_trans
    else:
        transA = cb_no_trans

    if A.ga.nd != 2:
        raise TypeError, "A is not a matrix"
    if transA == cb_no_trans:
        Yshp = A.ga.dimensions[0]
    else:
        Yshp = A.ga.dimensions[1]
    if Y is None:
        if beta != 0.0:
            raise ValueError, "Y not provided and beta != 0"
        Y = pygpu_empty(1, &Yshp, A.ga.typecode, GA_ANY_ORDER, A.context, None)
        overwrite_y = True

    if not overwrite_y:
        Y = pygpu_copy(Y, GA_ANY_ORDER)
    pygpu_blas_rgemv(transA, alpha, A, X, beta, Y, 0)

    return Y

def gemm(double alpha, GpuArray A, GpuArray B, double beta, GpuArray C=None,
         trans_a=False, trans_b=False, overwrite_c=False):
    cdef cb_transpose transA
    cdef cb_transpose transB
    cdef size_t[2] Cshp

    if trans_a:
        transA = cb_trans
    else:
        transA = cb_no_trans
    if trans_b:
        transB = cb_trans
    else:
        transB = cb_no_trans

    if A.ga.nd != 2:
        raise TypeError, "A is not a matrix"
    if B.ga.nd != 2:
        raise TypeError, "B is not a matrix"
    if transA == cb_no_trans:
        Cshp[0] = A.ga.dimensions[0]
    else:
        Cshp[0] = A.ga.dimensions[1]
    if transB == cb_no_trans:
        Cshp[1] = B.ga.dimensions[1]
    else:
        Cshp[1] = B.ga.dimensions[0]
    if C is None:
        if beta != 0.0:
            raise ValueError, "C not provided and beta != 0"
        C = pygpu_empty(2, Cshp, A.ga.typecode, GA_ANY_ORDER, A.context, None)
        overwrite_c = True

    if not overwrite_c:
        C = pygpu_copy(C, GA_ANY_ORDER)
    pygpu_blas_rgemm(transA, transB, alpha, A, B, beta, C, 0)

    return C

def ger(double alpha, GpuArray X, GpuArray Y, GpuArray A=None,
        overwrite_a=False):
    cdef size_t[2] Ashp

    if A is None:
        Ashp[0] = X.ga.dimensions[0];
        Ashp[1] = Y.ga.dimensions[0];
        A = pygpu_zeros(2, Ashp, X.ga.typecode, GA_ANY_ORDER, X.context, None)
        overwrite_a = True

    if not overwrite_a:
        A = pygpu_copy(A, GA_ANY_ORDER)
    pygpu_blas_rger(alpha, X, Y, A, 0)

    return A
