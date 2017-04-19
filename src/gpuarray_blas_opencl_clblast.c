#include "private.h"
#include "private_opencl.h"

#include "loaders/libclblast.h"

#include "gpuarray/buffer_blas.h"
#include "gpuarray/error.h"

static inline Layout convO(cb_order order) {
  switch (order) {
  case cb_row:
    return kRowMajor;
  case cb_column:
    return kColMajor;
  default:
    return -1;
  }
}

static inline Transpose convT(cb_transpose trans) {
  switch (trans) {
  case cb_no_trans:
    return kNo;
  case cb_trans:
    return kYes;
  case cb_conj_trans:
    return kConjugate;
  default:
    return -1;
  }
}

static const char *estr(CLBlastStatusCode err) {
  if (err > -1024)
    return cl_error_string((cl_int)err);
  switch (err) {
  case CLBlastNotImplemented:
    return "Unimplemented feature";
  case CLBlastInvalidMatrixA:
    return "matrix A is not a valid memory object";
  case CLBlastInvalidMatrixB:
    return "matrix B is not a valid memory object";
  case CLBlastInvalidMatrixC:
    return "matrix C is not a valid memory object";
  case CLBlastInvalidVectorX:
    return "vector X is not a valid memory object";
  case CLBlastInvalidVectorY:
    return "vector Y is not a valid memory object";
  case CLBlastInvalidDimension:
    return "An input dimension (M, N, K) is invalid";
  case CLBlastInvalidLeadDimA:
    return "leading dimension for A must not be less than the size of the first  dimension";
  case CLBlastInvalidLeadDimB:
    return "leading dimension for B must not be less than the size of the second dimension";
  case CLBlastInvalidLeadDimC:
    return "leading dimension for C must not be less than the size of the third dimension";
  case CLBlastInvalidIncrementX:
    return "increment for X must not be 0";
  case CLBlastInvalidIncrementY:
    return "increment for Y must not be 0";
  case CLBlastInsufficientMemoryA:
    return "memory object for matrix A is too small";
  case CLBlastInsufficientMemoryB:
    return "memory object for matrix B is too small";
  case CLBlastInsufficientMemoryC:
    return "memory object for matrix C is too small";
  case CLBlastInsufficientMemoryX:
    return "memory object for vector X is too small";
  case CLBlastInsufficientMemoryY:
    return "memory object for vector Y is too small";
  case CLBlastInvalidLocalMemUsage:
    return "not enough local memory on the device";
  case CLBlastNoHalfPrecision:
    return "float16 is not supported on this device";
  case CLBlastNoDoublePrecision:
    return "float64 is not supported on this device";
  case CLBlastInvalidVectorScalar:
    return "unit-sized vector is not a valid memory object";
  case CLBlastInsufficientMemoryScalar:
    return "memory object for unit-sized vector is too small";
  case CLBlastDatabaseError:
    return "device entry not in database";
  case CLBlastUnknownError:
    return "Unspecified error";
  case CLBlastUnexpectedError:
    return "Unexpected error";
  default:
    return "Unknow error";
  }
}

static inline int error_clblast(error *e, const char *msg,
                                CLBlastStatusCode err) {
  return error_fmt(e, GA_BLAS_ERROR, "%s: %s", msg, estr(err));
}

#define CLBT_CHECK(e, cmd) do {                 \
    CLBlastStatusCode err = (cmd);              \
    if (err != kSuccess)                        \
      return error_clblast(e, #cmd, err);       \
  } while (0)

static int setup(gpucontext *ctx) {
  return GA_NO_ERROR;
}

static void teardown(gpucontext *ctx) {
}

#define ARRAY_INIT(A)                           \
  if (A->ev != NULL)                            \
    clWaitForEvents(1, &A->ev)

#define ARRAY_FINI(A)                           \
  if (A->ev != NULL)                            \
    clReleaseEvent(A->ev);                      \
  A->ev = ev;                                   \
  clRetainEvent(A->ev)

static int hgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cl_ctx *ctx = A[0]->ctx;
  cl_event ev;
  size_t i;

  for (i = 0; i < batchCount; i++) {
    ARRAY_INIT(A[i]);
    ARRAY_INIT(B[i]);
    ARRAY_INIT(C[i]);
    CLBT_CHECK(ctx->err, CLBlastHgemm(convO(order), convT(transA),
                                      convT(transB), M, N, K,
                                      float_to_half(alpha),
                                      A[i]->buf, offA[i], lda,
                                      B[i]->buf, offB[i], ldb,
                                      float_to_half(beta),
                                      C[i]->buf, offC[i], ldc, &ctx->q, &ev));
    ARRAY_FINI(A[i]);
    ARRAY_FINI(B[i]);
    ARRAY_FINI(C[i]);
    clReleaseEvent(ev);
  }

  return GA_NO_ERROR;
}

static int sgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cl_ctx *ctx = A[0]->ctx;
  cl_event ev;
  size_t i;

  for (i = 0; i < batchCount; i++) {
    ARRAY_INIT(A[i]);
    ARRAY_INIT(B[i]);
    ARRAY_INIT(C[i]);
    CLBT_CHECK(ctx->err, CLBlastSgemm(convO(order), convT(transA),
                                      convT(transB), M, N, K,
                                      alpha, A[i]->buf, offA[i], lda,
                                      B[i]->buf, offB[i], ldb, beta,
                                      C[i]->buf, offC[i], ldc, &ctx->q, &ev));
    ARRAY_FINI(A[i]);
    ARRAY_FINI(B[i]);
    ARRAY_FINI(C[i]);
    clReleaseEvent(ev);
  }

  return GA_NO_ERROR;
}

static int dgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, double alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      double beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cl_ctx *ctx = A[0]->ctx;
  cl_event ev;
  size_t i;

  for (i = 0; i < batchCount; i++) {
    ARRAY_INIT(A[i]);
    ARRAY_INIT(B[i]);
    ARRAY_INIT(C[i]);
    CLBT_CHECK(ctx->err, CLBlastDgemm(convO(order), convT(transA),
                                      convT(transB), M, N, K,
                                      alpha, A[i]->buf, offA[i], lda,
                                      B[i]->buf, offB[i], ldb, beta,
                                      C[i]->buf, offC[i], ldc, &ctx->q, &ev));
    ARRAY_FINI(A[i]);
    ARRAY_FINI(B[i]);
    ARRAY_FINI(C[i]);
    clReleaseEvent(ev);
  }

  return GA_NO_ERROR;
}

static int hdot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(Z);

  CLBT_CHECK(ctx->err, CLBlastHdot(N, Z->buf, offZ, X->buf, offX, incX,
                                   Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(Z);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sdot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(Z);

  CLBT_CHECK(ctx->err, CLBlastSdot(N, Z->buf, offZ, X->buf, offX, incX,
                                   Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(Z);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int ddot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(Z);

  CLBT_CHECK(ctx->err, CLBlastDdot(N, Z->buf, offZ, X->buf, offX, incX,
                                   Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(Z);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int hgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX, float beta,
                 gpudata *Y, size_t offY, int incY) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(X);
  ARRAY_INIT(Y);

  CLBT_CHECK(ctx->err, CLBlastHgemv(convO(order), convT(transA), M, N,
                                    float_to_half(alpha),
                                    A->buf, offA, lda, X->buf, offX, incX,
                                    float_to_half(beta),
                                    Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(X);
  ARRAY_FINI(Y);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX, float beta,
                 gpudata *Y, size_t offY, int incY) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(X);
  ARRAY_INIT(Y);

  CLBT_CHECK(ctx->err, CLBlastSgemv(convO(order), convT(transA), M, N, alpha,
                                    A->buf, offA, lda, X->buf, offX, incX,
                                    beta, Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(X);
  ARRAY_FINI(Y);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int dgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 double alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX, double beta,
                 gpudata *Y, size_t offY, int incY) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(X);
  ARRAY_INIT(Y);

  CLBT_CHECK(ctx->err, CLBlastDgemv(convO(order), convT(transA), M, N, alpha,
                                    A->buf, offA, lda, X->buf, offX, incX,
                                    beta, Y->buf, offY, incY, &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(X);
  ARRAY_FINI(Y);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int hgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb, float beta,
                 gpudata *C, size_t offC, size_t ldc) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(B);
  ARRAY_INIT(C);

  CLBT_CHECK(ctx->err, CLBlastHgemm(convO(order), convT(transA), convT(transB),
                                    M, N, K, float_to_half(alpha),
                                    A->buf, offA, lda, B->buf, offB, ldb,
                                    float_to_half(beta), C->buf, offC, ldc,
                                    &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(B);
  ARRAY_FINI(C);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb, float beta,
                 gpudata *C, size_t offC, size_t ldc) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(B);
  ARRAY_INIT(C);

  CLBT_CHECK(ctx->err, CLBlastSgemm(convO(order), convT(transA), convT(transB),
                                    M, N, K, alpha,
                                    A->buf, offA, lda, B->buf, offB, ldb,
                                    beta, C->buf, offC, ldc, &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(B);
  ARRAY_FINI(C);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int dgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, double alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb, double beta,
                 gpudata *C, size_t offC, size_t ldc) {
  cl_ctx *ctx = A->ctx;
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(B);
  ARRAY_INIT(C);

  CLBT_CHECK(ctx->err, CLBlastDgemm(convO(order), convT(transA), convT(transB),
                                    M, N, K, alpha,
                                    A->buf, offA, lda, B->buf, offB, ldb,
                                    beta, C->buf, offC, ldc, &ctx->q, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(B);
  ARRAY_FINI(C);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int hger(cb_order order, size_t M, size_t N, float alpha,
                gpudata *X, size_t offX, int incX,
                gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(A);

  CLBT_CHECK(ctx->err, CLBlastHger(convO(order), M, N, float_to_half(alpha),
                                   X->buf, offX, incX, Y->buf, offY, incY,
                                   A->buf, offA, lda, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(A);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sger(cb_order order, size_t M, size_t N, float alpha,
                gpudata *X, size_t offX, int incX,
                gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(A);

  CLBT_CHECK(ctx->err, CLBlastSger(convO(order), M, N, alpha,
                                   X->buf, offX, incX, Y->buf, offY, incY,
                                   A->buf, offA, lda, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(A);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int dger(cb_order order, size_t M, size_t N, double alpha,
                gpudata *X, size_t offX, int incX,
                gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cl_ctx *ctx = X->ctx;
  cl_event ev;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(A);

  CLBT_CHECK(ctx->err, CLBlastDger(convO(order), M, N, alpha,
                                   X->buf, offX, incX, Y->buf, offY, incY,
                                   A->buf, offA, lda, &ctx->q, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(A);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

gpuarray_blas_ops clblast_ops = {
  setup,
  teardown,
  hdot,
  sdot,
  ddot,
  hgemv,
  sgemv,
  dgemv,
  hgemm,
  sgemm,
  dgemm,
  hger,
  sger,
  dger,
  hgemmBatch,
  sgemmBatch,
  dgemmBatch,
  NULL, /* hgemvBatch */
  NULL, /* sgemvBatch */
  NULL, /* dgemvBatch */
  NULL, /* hgerBatch */
  NULL, /* sgerBatch */
  NULL, /* dgerBatch */
};
