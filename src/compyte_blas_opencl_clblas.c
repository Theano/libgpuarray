#include "private.h"
#include "private_opencl.h"

#include <clBLAS.h>

#include "compyte/buffer_blas.h"
#include "compyte/error.h"

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

static int setup(void *c) {
  cl_ctx *ctx = (cl_ctx *)c;
  clblasStatus err;

  if (refcnt == 0) {
    err = clblasSetup();
    if (err != clblasSuccess)
      return GA_BLAS_ERROR;
  }

  if (ctx->blas_handle == NULL)
    ctx->blas_handle = &refcnt;
  refcnt++;
  return GA_NO_ERROR;
}

static void teardown(void *c) {
  cl_ctx *ctx = (cl_ctx *)c;
  if (ctx->blas_handle != NULL) {
    ctx->blas_handle = NULL;
    refcnt--;
  }
  if (refcnt == 0)
    clblasTeardown();
}

static int sgemv(const cb_order order,
                 const cb_transpose transA,
                 const size_t M,
                 const size_t N,
                 const float alpha,
                 gpudata *A,
                 const size_t offA,
                 const size_t lda,
                 gpudata *X,
                 const size_t offX,
                 const int incX,
                 const float beta,
                 gpudata *Y,
                 const size_t offY,
                 const int incY) {
  clblasStatus err;
  cl_uint num_ev = 0;
  cl_event evl[3];
  cl_event ev;
  if (A->ev != NULL)
    evl[num_ev++] = A->ev;
  if (X->ev != NULL)
    evl[num_ev++] = X->ev;
  if (Y->ev != NULL)
    evl[num_ev++] = Y->ev;
  err = clblasSgemv(convO(order), convT(transA), M, N, alpha, A->buf, offA,
                    lda, X->buf, offX, incX, beta, Y->buf, offY, incY,
                    1, &A->ctx->q, num_ev, num_ev == 0 ? NULL : evl, &ev);
  if (err != clblasSuccess)
    return GA_BLAS_ERROR;
  if (A->ev != NULL)
    clReleaseEvent(A->ev);
  A->ev = ev;
  clRetainEvent(ev);
  if (X->ev != NULL)
    clReleaseEvent(X->ev);
  X->ev = ev;
  clRetainEvent(ev);
  if (Y->ev != NULL)
    clReleaseEvent(Y->ev);
  Y->ev = ev;
  return GA_NO_ERROR;
}

COMPYTE_LOCAL const compyte_blas_ops clblas_ops = {
  setup,
  teardown,
  sgemv,
};
