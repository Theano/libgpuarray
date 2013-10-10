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
#define PREP_ORDER_GEMV				\
  cb_transpose rtransA = transA;		\
  size_t rN = N, rM = M, rlda = lda

#define HANDLE_ORDER_GEMV			       \
  if (order == cb_c) {				       \
    rM = N;					       \
    rN = M;					       \
    if (transA == cb_no_trans) {		       \
      rtransA = cb_trans;			       \
    } else {					       \
      rtransA = cb_no_trans;			       \
    }						       \
  }

#define PREP_ORDER_GEMM						\
  cb_transpose rtransA = transA, rtransB = transB;		\
  size_t rM = M, rN = N, rK = K, rlda = lda, rldb = ldb;	\
  gpudata *tmp;

#define HANDLE_ORDER_GEMM \
  if (order == cb_c) {	  \
    tmp = A;		  \
    A = B;		  \
    B = tmp;		  \
    rN = M;		  \
    rM = N;		  \
    rlda = ldb;		  \
    rldb = lda;		  \
  }

#define FUNC_INIT		\
  cuda_enter(ctx);		\
  if (ctx->err != CUDA_SUCCESS) \
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
#define TRANS(tr) convO(r ## tr)
#define SZ(s) r ## s
#define SCAL(s) &s
#define ARRAY(A, dtype) ((dtype *)A->ptr) + off ## A

#define POST_CALL			  \
  if (err == CUBLAS_STATUS_ARCH_MISMATCH) \
    return GA_DEVSUP_ERROR;		  \
  if (err != CUBLAS_STATUS_SUCCESS)	  \
    return GA_BLAS_ERROR

#include "generic_blas.inc.c"

