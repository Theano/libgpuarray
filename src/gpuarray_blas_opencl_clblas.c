#include "private.h"
#include "private_opencl.h"

#include "loaders/libclblas.h"

#include "gpuarray/buffer_blas.h"
#include "gpuarray/error.h"

extern const gpuarray_buffer_ops opencl_ops;

static inline clblasOrder convO(cb_order order) {
  switch (order) {
  case cb_row:
    return clblasRowMajor;
  case cb_column:
    return clblasColumnMajor;
  default:
    return -1;
  }
}

static inline clblasTranspose convT(cb_transpose trans) {
  switch (trans) {
  case cb_no_trans:
    return clblasNoTrans;
  case cb_trans:
    return clblasTrans;
  case cb_conj_trans:
    return clblasConjTrans;
  default:
    return -1;
  }
}

static unsigned int refcnt = 0;

static const char *estr(clblasStatus err) {
  if (err > -1024)
    return cl_error_string((cl_int)err);
  switch (err) {
  case clblasNotImplemented:
    return "Unimplemented feature";
  case clblasNotInitialized:
    return "Library not initialized";
  case clblasInvalidMatA:
    return "matrix A is not a valid memory object";
  case clblasInvalidMatB:
    return "matrix B is not a valid memory object";
  case clblasInvalidMatC:
    return "matrix C is not a valid memory object";
  case clblasInvalidVecX:
    return "vector X is not a valid memory object";
  case clblasInvalidVecY:
    return "vector Y is not a valid memory object";
  case clblasInvalidDim:
    return "An input dimension (M, N, K) is invalid";
  case clblasInvalidLeadDimA:
    return "leading dimension for A must not be less than the size of the first dimension";
  case clblasInvalidLeadDimB:
    return "leading dimension for B must not be less than the size of the second dimension";
  case clblasInvalidLeadDimC:
    return "leading dimension for C must not be less than the size of the third dimension";
  case clblasInvalidIncX:
    return "increment for X must not be 0";
  case clblasInvalidIncY:
    return "increment for Y must not be 0";
  case clblasInsufficientMemMatA:
    return "memory object for matrix A is too small";
  case clblasInsufficientMemMatB:
    return "memory object for matrix B is too small";
  case clblasInsufficientMemMatC:
    return "memory object for matrix C is too small";
  case clblasInsufficientMemVecX:
    return "memory object for vector X is too small";
  case clblasInsufficientMemVecY:
    return "memory object for vector Y is too small";
  default:
    return "Unknow error";
  }
}

static inline int error_clblas(error *e, const char *msg, clblasStatus err) {
  return error_fmt(e, GA_BLAS_ERROR, "%s: %s", msg, estr(err));
}

#define CLB_CHECK(e, cmd) do {                  \
    clblasStatus err = (cmd);                   \
    if (err != clblasSuccess)                   \
      return error_clblas(e, #cmd, err);        \
  } while (0)

static int setup(gpucontext *ctx) {
  if (refcnt == 0) {
    CLB_CHECK(ctx->err, clblasSetup());
  }

  if (ctx->blas_handle == NULL)
    ctx->blas_handle = &refcnt;
  refcnt++;
  return GA_NO_ERROR;
}

static void teardown(gpucontext *ctx) {
  if (ctx->blas_handle != NULL) {
    ctx->blas_handle = NULL;
    refcnt--;
  }
  if (refcnt == 0)
    clblasTeardown();
}

#define ARRAY_INIT(A)                           \
  if (A->ev != NULL)                            \
    evl[num_ev++] = A->ev

#define ARRAY_FINI(A)                           \
  if (A->ev != NULL)                            \
    clReleaseEvent(A->ev);                      \
  A->ev = ev;                                   \
  clRetainEvent(A->ev)

static int sgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cl_ctx *ctx = A[0]->ctx;
  cl_event evl[3];
  cl_event ev;
  size_t i;
  cl_uint num_ev = 0;

  for (i = 0; i < batchCount; i++) {
    ARRAY_INIT(A[i]);
    ARRAY_INIT(B[i]);
    ARRAY_INIT(C[i]);
    CLB_CHECK(ctx->err, clblasSgemm(convO(order), convT(transA), convT(transB),
                                    M, N, K,
                                    alpha, A[i]->buf, offA[i], lda,
                                    B[i]->buf, offB[i], ldb,
                                    beta, C[i]->buf, offC[i], ldc, 1, &ctx->q,
                                    num_ev, num_ev == 0 ? NULL : evl, &ev));
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
  cl_event evl[3];
  cl_event ev;
  size_t i;
  cl_uint num_ev = 0;

  for (i = 0; i < batchCount; i++) {
    ARRAY_INIT(A[i]);
    ARRAY_INIT(B[i]);
    ARRAY_INIT(C[i]);
    CLB_CHECK(ctx->err, clblasDgemm(convO(order), convT(transA), convT(transB),
                                    M, N, K,
                                    alpha, A[i]->buf, offA[i], lda,
                                    B[i]->buf, offB[i], ldb,
                                    beta, C[i]->buf, offC[i], ldc, 1, &ctx->q,
                                    num_ev, num_ev == 0 ? NULL : evl, &ev));
    ARRAY_FINI(A[i]);
    ARRAY_FINI(B[i]);
    ARRAY_FINI(C[i]);
    clReleaseEvent(ev);
  }

  return GA_NO_ERROR;
}

static int sdot(
        size_t N,
        gpudata *X, size_t offX, size_t incX,
        gpudata *Y, size_t offY, size_t incY,
        gpudata *Z, size_t offZ) {
  cl_ctx *ctx = X->ctx;
  clblasStatus err;
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;
  gpudata *wbuf;

  wbuf = opencl_ops.buffer_alloc((gpucontext*)ctx,
                                 N*sizeof(float), NULL, GA_BUFFER_READ_WRITE);
  if (wbuf == NULL)
      return ctx->err->code;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(Z);

  // TODO: a thread-safe static buffer or allocator?
  err = clblasSdot(
          N, Z->buf, offZ,
          X->buf, offX, incX,
          Y->buf, offY, incY,
          wbuf->buf, 1, &ctx->q,
          num_ev, num_ev ? evl : NULL, &ev);
  opencl_ops.buffer_release(wbuf);
  if (err != clblasSuccess)
    return error_clblas(ctx->err, "clblasSdot", err);

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
  clblasStatus err;
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;
  gpudata *wbuf;

  wbuf = opencl_ops.buffer_alloc((gpucontext*)ctx,
                                 N*sizeof(double), NULL, GA_BUFFER_READ_WRITE);
  if (wbuf == NULL)
      return ctx->err->code;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(Z);

  err = clblasDdot(
          N, Z->buf, offZ,
          X->buf, offX, incX,
          Y->buf, offY, incY,
          wbuf->buf, 1, &ctx->q,
          num_ev, num_ev ? evl : NULL, &ev);
  opencl_ops.buffer_release(wbuf);
  if (err != clblasSuccess)
    return error_clblas(ctx->err, "clblasDdot", err);

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(Z);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX, float beta,
                 gpudata *Y, size_t offY, int incY) {
  cl_ctx *ctx = A->ctx;
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(X);
  ARRAY_INIT(Y);

  CLB_CHECK(ctx->err, clblasSgemv(convO(order), convT(transA), M, N, alpha,
                                  A->buf, offA, lda, X->buf, offX, incX,
                                  beta, Y->buf, offY, incY, 1, &ctx->q,
                                  num_ev, num_ev == 0 ? NULL : evl, &ev));

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
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(X);
  ARRAY_INIT(Y);

  CLB_CHECK(ctx->err, clblasDgemv(convO(order), convT(transA), M, N, alpha,
                                  A->buf, offA, lda, X->buf, offX, incX,
                                  beta, Y->buf, offY, incY, 1, &ctx->q,
                                  num_ev, num_ev == 0 ? NULL : evl, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(X);
  ARRAY_FINI(Y);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb, float beta,
                 gpudata *C, size_t offC, size_t ldc) {
  cl_ctx *ctx = A->ctx;
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(B);
  ARRAY_INIT(C);

  CLB_CHECK(ctx->err, clblasSgemm(convO(order), convT(transA), convT(transB),
                                  M, N, K,
                                  alpha, A->buf, offA, lda, B->buf, offB, ldb,
                                  beta, C->buf, offC, ldc, 1, &ctx->q,
                                  num_ev, num_ev == 0 ? NULL : evl, &ev));

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
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;

  ARRAY_INIT(A);
  ARRAY_INIT(B);
  ARRAY_INIT(C);

  CLB_CHECK(ctx->err, clblasDgemm(convO(order), convT(transA), convT(transB),
                                  M, N, K,
                                  alpha, A->buf, offA, lda, B->buf, offB, ldb,
                                  beta, C->buf, offC, ldc, 1, &ctx->q,
                                  num_ev, num_ev == 0 ? NULL : evl, &ev));

  ARRAY_FINI(A);
  ARRAY_FINI(B);
  ARRAY_FINI(C);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int sger(cb_order order, size_t M, size_t N, float alpha,
                gpudata *X, size_t offX, int incX,
                gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cl_ctx *ctx = X->ctx;
  cl_event evl[3];
  cl_event ev;
  cl_uint num_ev = 0;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(A);

  CLB_CHECK(ctx->err, clblasSger(convO(order), M, N, alpha, X->buf, offX, incX,
                                 Y->buf, offY, incY, A->buf, offA, lda, 1, &ctx->q,
                                 num_ev, num_ev == 0 ? NULL : evl, &ev));

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
  cl_event evl[3];
  cl_event ev;
  cl_uint num_ev = 0;

  ARRAY_INIT(X);
  ARRAY_INIT(Y);
  ARRAY_INIT(A);

  CLB_CHECK(ctx->err, clblasDger(convO(order), M, N, alpha, X->buf, offX, incX,
                                 Y->buf, offY, incY, A->buf, offA, lda, 1, &ctx->q,
                                 num_ev, num_ev == 0 ? NULL : evl, &ev));

  ARRAY_FINI(X);
  ARRAY_FINI(Y);
  ARRAY_FINI(A);

  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

gpuarray_blas_ops clblas_ops = {
  setup,
  teardown,
  NULL, /* hdot */
  sdot,
  ddot,
  NULL, /* hgemv */
  sgemv,
  dgemv,
  NULL, /* hgemm */
  sgemm,
  dgemm,
  NULL, /* hger */
  sger,
  dger,
  NULL, /* hgemmBatch */
  sgemmBatch,
  dgemmBatch,
  NULL, /* hgemvBatch */
  NULL, /* sgemvBatch */
  NULL, /* dgemvBatch */
  NULL, /* hgerBatch */
  NULL, /* sgerBatch */
  NULL, /* dgerBatch */
};
