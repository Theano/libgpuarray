#ifndef GPUARRAY_BUFFER_BLAS_H
#define GPUARRAY_BUFFER_BLAS_H

#include <gpuarray/buffer.h>
#include <gpuarray/config.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _cb_order {
  cb_row,
  cb_column
} cb_order;

#define cb_c cb_row
#define cb_fortran cb_column

typedef enum _cb_side {
  cb_left,
  cb_right
} cb_side;

typedef enum _cb_transpose {
  cb_no_trans,
  cb_trans,
  cb_conj_trans
} cb_transpose;

typedef enum _cb_uplo {
  cb_upper,
  cb_lower
} cb_uplo;

typedef struct _gpuarray_blas_ops {
  int (*setup)(gpucontext *ctx);
  void (*teardown)(gpucontext *ctx);
  int (*hgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               float alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, float beta,
               gpudata *Y, size_t offY, int incY);
  int (*sgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               float alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, float beta,
               gpudata *Y, size_t offY, int incY);
  int (*dgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               double alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, double beta,
               gpudata *Y, size_t offY, int incY);
  int (*hgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, float alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               float beta, gpudata *C, size_t offC, size_t ldc);
  int (*sgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, float alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               float beta, gpudata *C, size_t offC, size_t ldc);
  int (*dgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, double alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               double beta, gpudata *C, size_t offC, size_t ldc);
  int (*hger)(cb_order order, size_t M, size_t N, float alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*sger)(cb_order order, size_t M, size_t N, float alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*dger)(cb_order order, size_t M, size_t N, double alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*hgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    float beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*sgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    float beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*dgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, double alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    double beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*hgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    float beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*sgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    float beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*dgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, double alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    double beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*hgerBatch)(cb_order order, size_t M, size_t N, float alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
  int (*sgerBatch)(cb_order order, size_t M, size_t N, float alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
  int (*dgerBatch)(cb_order order, size_t M, size_t N, double alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
} gpuarray_blas_ops;

#ifdef __cplusplus
}
#endif

#endif
