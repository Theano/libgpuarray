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

GPUARRAY_PUBLIC int gpublas_setup(gpucontext *ctx);

GPUARRAY_PUBLIC void gpublas_teardown(gpucontext *ctx);

GPUARRAY_PUBLIC const char *gpublas_error(gpucontext *ctx);

GPUARRAY_PUBLIC int gpublas_hgemv(
  cb_order order, cb_transpose transA, size_t M, size_t N, float alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *X, size_t offX, int incX,
  float beta, gpudata *Y, size_t offY, int incY);

GPUARRAY_PUBLIC int gpublas_sgemv(
  cb_order order, cb_transpose transA, size_t M, size_t N, float alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *X, size_t offX, int incX,
  float beta, gpudata *Y, size_t offY, int incY);

GPUARRAY_PUBLIC int gpublas_dgemv(
  cb_order order, cb_transpose transA, size_t M, size_t N, double alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *X, size_t offX, int incX,
  double beta, gpudata *Y, size_t offY, int incY);

GPUARRAY_PUBLIC int gpublas_hgemm(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *B, size_t offB, size_t ldb,
  float beta, gpudata *C, size_t offC, size_t ldc);

GPUARRAY_PUBLIC int gpublas_sgemm(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *B, size_t offB, size_t ldb,
  float beta, gpudata *C, size_t offC, size_t ldc);

GPUARRAY_PUBLIC int gpublas_dgemm(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, double alpha,
  gpudata *A, size_t offA, size_t lda, gpudata *B, size_t offB, size_t ldb,
  double beta, gpudata *C, size_t offC, size_t ldc);

GPUARRAY_PUBLIC int gpublas_hger(
  cb_order order, size_t M, size_t N, float alpha,
  gpudata *X, size_t offX, int incX,
  gpudata *Y, size_t offY, int incY,
  gpudata *A, size_t offA, size_t lda);

GPUARRAY_PUBLIC int gpublas_sger(
  cb_order order, size_t M, size_t N, float alpha,
  gpudata *X, size_t offX, int incX,
  gpudata *Y, size_t offY, int incY,
  gpudata *A, size_t offA, size_t lda);

GPUARRAY_PUBLIC int gpublas_dger(
  cb_order order, size_t M, size_t N, double alpha,
  gpudata *X, size_t offX, int incX,
  gpudata *Y, size_t offY, int incY,
  gpudata *A, size_t offA, size_t lda);

GPUARRAY_PUBLIC int gpublas_hgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  float beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_sgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  float beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_dgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  double beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_hgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_sgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_dgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  double beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_hgerBatch(
  cb_order order, size_t M, size_t N, float alpha,
  gpudata **x, size_t *offX, size_t incX,
  gpudata **y, size_t *offY, size_t incY,
  gpudata **A, size_t *offA, size_t lda,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_sgerBatch(
  cb_order order, size_t M, size_t N, float alpha,
  gpudata **x, size_t *offX, size_t incX,
  gpudata **y, size_t *offY, size_t incY,
  gpudata **A, size_t *offA, size_t lda,
  size_t batchCount, int flags);

GPUARRAY_PUBLIC int gpublas_dgerBatch(
  cb_order order, size_t M, size_t N, double alpha,
  gpudata **x, size_t *offX, size_t incX,
  gpudata **y, size_t *offY, size_t incY,
  gpudata **A, size_t *offA, size_t lda,
  size_t batchCount, int flags);

#ifdef __cplusplus
}
#endif

#endif
