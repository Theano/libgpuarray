#include "private.h"

#include <gpuarray/error.h>

int gpublas_setup(gpucontext *ctx) {
  if (ctx->blas_ops == NULL)
    return GA_UNSUPPORTED_ERROR;
  return ctx->blas_ops->setup(ctx);
}

void gpublas_teardown(gpucontext *ctx) {
  if (ctx->blas_ops != NULL)
    return ctx->blas_ops->teardown(ctx);
}

const char *gpublas_error(gpucontext *ctx) {
  if (ctx->blas_ops != NULL)
    return ctx->blas_ops->error(ctx);
  return "No blas ops available, API error.";
}

int gpublas_hgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  float beta,
                  gpudata *Y, size_t offY, int incY) {
  return gpudata_context(A)->blas_ops->hgemv(
    order, transA, M, N, alpha, A, offA, lda,
    X, offX, incX, beta, Y, offY, incY);
}

int gpublas_sgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  float beta,
                  gpudata *Y, size_t offY, int incY) {
  return gpudata_context(A)->blas_ops->sgemv(
    order, transA, M, N, alpha, A, offA, lda,
    X, offX, incX, beta, Y, offY, incY);
}

int gpublas_dgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, double alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  double beta,
                  gpudata *Y, size_t offY, int incY) {
  return gpudata_context(A)->blas_ops->dgemv(
    order, transA, M, N, alpha, A, offA, lda,
    X, offX, incX, beta, Y, offY, incY);
}

int gpublas_hgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  float beta, gpudata *C, size_t offC, size_t ldc) {
  return gpudata_context(A)->blas_ops->hgemm(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc);
}

int gpublas_sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  float beta, gpudata *C, size_t offC, size_t ldc) {
  return gpudata_context(A)->blas_ops->sgemm(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc);
}

int gpublas_dgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, double alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  double beta, gpudata *C, size_t offC, size_t ldc) {
  return gpudata_context(A)->blas_ops->dgemm(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc);
}

int gpublas_hger(cb_order order, size_t M, size_t N, float alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  return gpudata_context(X)->blas_ops->hger(
    order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda);
}

int gpublas_sger(cb_order order, size_t M, size_t N, float alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  return gpudata_context(X)->blas_ops->sger(
    order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda);
}

int gpublas_dger(cb_order order, size_t M, size_t N, double alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  return gpudata_context(X)->blas_ops->dger(
    order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda);
}

int gpublas_hgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  float beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags) {
  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->hgemmBatch(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc, batchCount);
}

int gpublas_sgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  float beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags) {
  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->sgemmBatch(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc, batchCount);
}

int gpublas_dgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  double beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags) {
  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->dgemmBatch(
    order, transA, transB, M, N, K, alpha, A, offA, lda,
    B, offB, ldb, beta, C, offC, ldc, batchCount);
}

int gpublas_hgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->hgemvBatch(
    order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
    beta, y, offY, incY, batchCount, flags);
}

int gpublas_sgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->sgemvBatch(
    order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
    beta, y, offY, incY, batchCount, flags);
}

int gpublas_dgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  double beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(A[0])->blas_ops->dgemvBatch(
    order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
    beta, y, offY, incY, batchCount, flags);
}

int gpublas_hgerBatch(cb_order order, size_t M, size_t N, float alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(x[0])->blas_ops->hgerBatch(
    order, M, N, alpha, x, offX, incX, y, offY, incY,
    A, offA, lda, batchCount, flags);
}

int gpublas_sgerBatch(cb_order order, size_t M, size_t N, float alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(x[0])->blas_ops->sgerBatch(
    order, M, N, alpha, x, offX, incX, y, offY, incY,
    A, offA, lda, batchCount, flags);
}

int gpublas_dgerBatch(cb_order order, size_t M, size_t N, double alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  if (batchCount == 0) return GA_NO_ERROR;
  return gpudata_context(x[0])->blas_ops->dgerBatch(
    order, M, N, alpha, x, offX, incX, y, offY, incY,
    A, offA, lda, batchCount, flags);
}
