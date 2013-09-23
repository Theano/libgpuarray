#include "private.h"
#include "private_cuda.h"

#include "compyte/blas.h"
#include "compyte/error.h"

#include "cublas_v2.h"

static inline cublasOperation_t convO(cb_transpose order) {
  switch (order) {
  case cb_no_trans:
    return CUBLAS_OP_N;
  case cb_trans:
    return CUBLAS_OP_T;
  case cb_conj_trans:
    return CUBLAS_OP_C;
  default:
    return -1;
  }
}

static int setup(void *c) {
  cuda_context *ctx = (cuda_context *)c;
  cublasHandle_t handle;
  cublasStatus_t err;

  if (ctx->blas_handle != NULL)
    return GA_NO_ERROR;

  cuda_enter(ctx);
  err = cublasCreate(&handle);
  cuda_exit(ctx);

  if (err != CUBLAS_STATUS_SUCCESS)
    return GA_BLAS_ERROR;

  err = cublasSetStream(handle, ctx->s);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    return GA_BLAS_ERROR;
  }

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);

  ctx->blas_handle = handle;

  return GA_NO_ERROR;
}

static void teardown(void *c) {
  cuda_context *ctx = (cuda_context *)c;

  if (ctx->blas_handle == NULL)
    return;

  cuda_enter(ctx);
  cublasDestroy(ctx->blas_handle);
  ctx->blas_handle = NULL;
  cuda_exit(ctx);
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
  cublasStatus_t err;
  cuda_context *ctx = A->ctx;
  cuda_enter(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;

  ctx->err = cuStreamWaitEvent(ctx->s, A->ev, 0);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    return GA_IMPL_ERROR;
  }
  ctx->err = cuStreamWaitEvent(ctx->s, X->ev, 0);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    return GA_IMPL_ERROR;
  }
  ctx->err = cuStreamWaitEvent(ctx->s, Y->ev, 0);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    return GA_IMPL_ERROR;
  }

  err = cublasSgemv(ctx->blas_handle, convO(transA), M, N, &alpha,
		    ((float *)A->ptr) + offA, lda, 
		    ((float *)X->ptr) + offX, incX, &beta,
		    ((float *)Y->ptr) + offY, incY);
  if (err == CUBLAS_STATUS_ARCH_MISMATCH)
    return GA_DEVSUP_ERROR;
  if (err != CUBLAS_STATUS_SUCCESS)
    return GA_BLAS_ERROR;

  cuEventRecord(A->ev, ctx->s);
  cuEventRecord(X->ev, ctx->s);
  cuEventRecord(Y->ev, ctx->s);
  return GA_NO_ERROR;
}

COMPYTE_LOCAL compyte_blas_ops cublas_ops = {
  setup,
  teardown,
  sgemv
};
