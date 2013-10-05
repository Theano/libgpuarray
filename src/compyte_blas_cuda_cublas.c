#include "private.h"
#include "private_cuda.h"

#include "compyte/buffer_blas.h"
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
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;
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

#define NAME cublas

#define FETCH_CONTEXT(A) cuda_context *ctx = (A)->ctx
#define FUNC_DECLS cublasStatus_t err
#define PREP_ORDER1(transA, M, N, A, lda) \
  cb_transpose r ## transA = transA; \
  size_t r ## N = N, r ## M = M, r ## lda = lda

#define HANDLE_ORDER1(order, transA, M, N, A, lda) \
  if (order == cb_c) {				   \
    r ## M = N;					   \
    r ## N = M;					   \
    r ## lda = r ## M;				   \
    if (transA == cb_no_trans) {		   \
      r ## transA = cb_trans;			   \
    } else if (transA == cb_trans) {		   \
      r ## transA = cb_no_trans;		   \
    } else {					   \
      return GA_DEVSUP_ERROR;			   \
    }						   \
  }

#define FUNC_INIT	       \
  cuda_enter(ctx);	       \
  if (ctx->err != GA_NO_ERROR) \
    return GA_IMPL_ERROR

#define ARRAY_INIT(A)				  \
  ctx->err = cuStreamWaitEvent(ctx->s, A->ev, 0); \
  if (ctx->err != CUDA_SUCCESS) {		  \
    cuda_exit(ctx);				  \
    return GA_IMPL_ERROR;			  \
  }
#define ARRAY_FINI(A) cuEventRecord(A->ev, ctx->s)

#define PRE_CALL err =
#define PREFIX(typec, TYPEC) cublas ## TYPEC
#define INIT_ARGS ctx->blas_handle,
#define TRANS(tr) convO(tr)
#define SZ(s) r ## s
#define SCAL(s) &s
#define ARRAY(A, dtype) ((dtype *)A->ptr) + off ## A

#define POST_CALL			  \
  if (err == CUBLAS_STATUS_ARCH_MISMATCH) \
    return GA_DEVSUP_ERROR;		  \
  if (err != CUBLAS_STATUS_SUCCESS)	  \
    return GA_BLAS_ERROR

#include "generic_blas.inc.c"

