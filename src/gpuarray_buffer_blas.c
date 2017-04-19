#include "private.h"

#include <gpuarray/error.h>

int gpublas_setup(gpucontext *ctx) {
  if (ctx->blas_ops == NULL)
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Missing Blas library");
  return ctx->blas_ops->setup(ctx);
}

void gpublas_teardown(gpucontext *ctx) {
  if (ctx->blas_ops != NULL)
    ctx->blas_ops->teardown(ctx);
}

const char *gpublas_error(gpucontext *ctx) {
  return ctx->err->msg;
}

#define BLAS_OP(buf,name, args)                                         \
  gpucontext *ctx = gpudata_context(buf);                               \
  if (ctx->blas_ops->name)                                              \
    return ctx->blas_ops->name args;                                    \
  else                                                                  \
    return error_fmt(ctx->err, GA_DEVSUP_ERROR, "Blas operation not supported by device or missing library: %s", #name)


int gpublas_hdot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  BLAS_OP(X, hdot, (N, X, offX, incX, Y, offY, incY, Z, offZ));
}

int gpublas_sdot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  BLAS_OP(X, sdot, (N, X, offX, incX, Y, offY, incY, Z, offZ));
}

int gpublas_ddot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  BLAS_OP(X, ddot, (N, X, offX, incX, Y, offY, incY, Z, offZ));
}

int gpublas_hgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  float beta,
                  gpudata *Y, size_t offY, int incY) {
  BLAS_OP(A, hgemv, (order, transA, M, N, alpha, A, offA, lda,
                     X, offX, incX, beta, Y, offY, incY));
}

int gpublas_sgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  float beta,
                  gpudata *Y, size_t offY, int incY) {
  BLAS_OP(A, sgemv, (order, transA, M, N, alpha, A, offA, lda,
                     X, offX, incX, beta, Y, offY, incY));
}

int gpublas_dgemv(cb_order order, cb_transpose transA,
                  size_t M, size_t N, double alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *X, size_t offX, int incX,
                  double beta,
                  gpudata *Y, size_t offY, int incY) {
  BLAS_OP(A, dgemv, (order, transA, M, N, alpha, A, offA, lda,
                     X, offX, incX, beta, Y, offY, incY));
}

int gpublas_hgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  float beta, gpudata *C, size_t offC, size_t ldc) {
  BLAS_OP(A, hgemm, (order, transA, transB, M, N, K, alpha, A, offA, lda,
                     B, offB, ldb, beta, C, offC, ldc));
}

int gpublas_sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, float alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  float beta, gpudata *C, size_t offC, size_t ldc) {
  BLAS_OP(A, sgemm, (order, transA, transB, M, N, K, alpha, A, offA, lda,
                     B, offB, ldb, beta, C, offC, ldc));
}

int gpublas_dgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                  size_t M, size_t N, size_t K, double alpha,
                  gpudata *A, size_t offA, size_t lda,
                  gpudata *B, size_t offB, size_t ldb,
                  double beta, gpudata *C, size_t offC, size_t ldc) {
  BLAS_OP(A, dgemm, (order, transA, transB, M, N, K, alpha, A, offA, lda,
                     B, offB, ldb, beta, C, offC, ldc));
}

int gpublas_hger(cb_order order, size_t M, size_t N, float alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  BLAS_OP(X, hger,
          (order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda));
}

int gpublas_sger(cb_order order, size_t M, size_t N, float alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  BLAS_OP(X, sger,
          (order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda));
}

int gpublas_dger(cb_order order, size_t M, size_t N, double alpha,
                 gpudata *X, size_t offX, int incX,
                 gpudata *Y, size_t offY, int incY,
                 gpudata *A, size_t offA, size_t lda) {
  BLAS_OP(X, dger,
          (order, M, N, alpha, X, offX, incX, Y, offY, incY, A, offA, lda));
}

#define BLAS_OPB(l, name, args)                                         \
  gpucontext *ctx;                                                      \
  if (batchCount == 0) return GA_NO_ERROR;                              \
  ctx = gpudata_context(l[0]);                                          \
  if (ctx->blas_ops->name)                                              \
    return ctx->blas_ops->name args;                                    \
  else                                                                  \
    return error_fmt(ctx->err, GA_DEVSUP_ERROR, "Blas operation not supported by library in use: %s", #name)

#define BLAS_OPBF(l, name, args)                                        \
  gpucontext *ctx;                                                      \
  if (batchCount == 0) return GA_NO_ERROR;                              \
  ctx = gpudata_context(l[0]);                                          \
  if (flags != 0) return error_set(ctx->err, GA_INVALID_ERROR, "flags is not 0"); \
  if (ctx->blas_ops->name)                                              \
    return ctx->blas_ops->name args;                                    \
  else                                                                  \
    return error_fmt(ctx->err, GA_DEVSUP_ERROR, "Blas operation not supported by library in use: %s", #name)

int gpublas_hgemmBatch(
    cb_order order, cb_transpose transA, cb_transpose transB,
    size_t M, size_t N, size_t K, float alpha,
    gpudata **A, size_t *offA, size_t lda,
    gpudata **B, size_t *offB, size_t ldb,
    float beta, gpudata **C, size_t *offC, size_t ldc,
    size_t batchCount, int flags) {
  BLAS_OPBF(A, hgemmBatch,
            (order, transA, transB, M, N, K, alpha, A, offA, lda,
             B, offB, ldb, beta, C, offC, ldc, batchCount));
}

int gpublas_sgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  float beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags) {
  BLAS_OPBF(A, sgemmBatch,
            (order, transA, transB, M, N, K, alpha, A, offA, lda,
             B, offB, ldb, beta, C, offC, ldc, batchCount));
}

int gpublas_dgemmBatch(
  cb_order order, cb_transpose transA, cb_transpose transB,
  size_t M, size_t N, size_t K, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **B, size_t *offB, size_t ldb,
  double beta, gpudata **C, size_t *offC, size_t ldc,
  size_t batchCount, int flags) {
  BLAS_OPBF(A, dgemmBatch,
            (order, transA, transB, M, N, K, alpha, A, offA, lda,
             B, offB, ldb, beta, C, offC, ldc, batchCount));
}

int gpublas_hgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  BLAS_OPB(A, hgemvBatch,
           (order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
            beta, y, offY, incY, batchCount, flags));
}

int gpublas_sgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, float alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  float beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  BLAS_OPB(A, sgemvBatch,
           (order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
            beta, y, offY, incY, batchCount, flags));
}

int gpublas_dgemvBatch(
  cb_order order, cb_transpose transA,
  size_t M, size_t N, double alpha,
  gpudata **A, size_t *offA, size_t lda,
  gpudata **x, size_t *offX, size_t incX,
  double beta, gpudata **y, size_t *offY, size_t incY,
  size_t batchCount, int flags) {
  BLAS_OPB(A, dgemvBatch,
           (order, transA, M, N, alpha, A, offA, lda, x, offX, incX,
            beta, y, offY, incY, batchCount, flags));
}

int gpublas_hgerBatch(cb_order order, size_t M, size_t N, float alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  BLAS_OPB(x, hgerBatch,
           (order, M, N, alpha, x, offX, incX, y, offY, incY,
            A, offA, lda, batchCount, flags));
}

int gpublas_sgerBatch(cb_order order, size_t M, size_t N, float alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  BLAS_OPB(x, sgerBatch,
           (order, M, N, alpha, x, offX, incX, y, offY, incY,
            A, offA, lda, batchCount, flags));
}

int gpublas_dgerBatch(cb_order order, size_t M, size_t N, double alpha,
                      gpudata **x, size_t *offX, size_t incX,
                      gpudata **y, size_t *offY, size_t incY,
                      gpudata **A, size_t *offA, size_t lda,
                      size_t batchCount, int flags) {
  BLAS_OPB(x, dgerBatch,
           (order, M, N, alpha, x, offX, incX, y, offY, incY,
            A, offA, lda, batchCount, flags));
}
