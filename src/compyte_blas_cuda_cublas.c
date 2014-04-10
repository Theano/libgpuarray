#include "private.h"
#include "private_cuda.h"

#include "compyte/buffer_blas.h"
#include "compyte/error.h"

#include "cublas_v2.h"

static inline cublasOperation_t convT(cb_transpose trans) {
  switch (trans) {
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
#define PREP_ORDER_GEMV size_t t

#define HANDLE_ORDER_GEMV			       \
  if (order == cb_c) {				       \
    t = N;					       \
    N = M;					       \
    M = t;					       \
    if (transA == cb_no_trans) {		       \
      transA = cb_trans;			       \
    } else {					       \
      transA = cb_no_trans;			       \
    }						       \
  }

#define PREP_ORDER_GEMM							\
  size_t lt, t;								\
  gpudata *T;								\
  cb_transpose transT

#define PREP_ORDER_GEMMBATCH						\
  size_t *lt, t;                                                        \
  gpudata **T;								\
  cb_transpose transT

#define HANDLE_ORDER_GEMM			       \
  if (order == cb_c) {				       \
    t = N;					       \
    N = M;					       \
    M = t;					       \
    T = A;					       \
    A = B;					       \
    B = T;					       \
    t = lda;					       \
    lda = ldb;					       \
    ldb = t;					       \
    transT = transA;				       \
    transA = transB;				       \
    transB = transT;				       \
    lt = offA;					       \
    offA = offB;				       \
    offB = lt;					       \
  }


#define HANDLE_ORDER_GEMMBATCH HANDLE_ORDER_GEMM

#define PREP_ORDER_GER \
  size_t t;	       \
  gpudata *td

#define HANDLE_ORDER_GER \
  if (order == cb_c) {	 \
    t = M;		 \
    M = N;		 \
    N = t;		 \
    t = offX;		 \
    offX = offY;	 \
    offY = t;		 \
    t = incX;		 \
    incX = incY;	 \
    incY = t;		 \
    td = X;		 \
    X = Y;		 \
    Y = td;		 \
  }


#define FUNC_INIT		\
  cuda_enter(ctx);		\
  if (ctx->err != CUDA_SUCCESS) \
    return GA_IMPL_ERROR

#define FUNC_FINI cuda_exit(ctx)

/*#define ARRAY_INIT(A)				  \
  ctx->err = cuStreamWaitEvent(ctx->s, (A)->ev, 0); \
  if (ctx->err != CUDA_SUCCESS) {		  \
    cuda_exit(ctx);				  \
    return GA_IMPL_ERROR;			  \
    }*/
#define ARRAY_INIT(A)

/*#define ARRAY_FINI(A) cuEventRecord((A)->ev, ctx->s)*/
#define ARRAY_FINI(A)

#define PRE_CALL err =
#define PREFIX(typec, TYPEC) cublas ## TYPEC
#define INIT_ARGS ctx->blas_handle,
#define TRANS(tr) convT(tr)
#define SZ(s) s
#define SCAL(s) &s
#define ARRAY(A, dtype) ((dtype *)A->ptr) + off ## A

#define POST_CALL			  \
  if (err == CUBLAS_STATUS_ARCH_MISMATCH) \
    return GA_DEVSUP_ERROR;		  \
  if (err != CUBLAS_STATUS_SUCCESS)	  \
    return GA_BLAS_ERROR

static int sgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  FETCH_CONTEXT(A[0]);
  FUNC_DECLS;
  PREP_ORDER_GEMMBATCH;
  const float **A_l = alloca(sizeof(float *) * batchCount);
  const float **B_l = alloca(sizeof(float *) * batchCount);
  float **C_l = alloca(sizeof(float *) * batchCount);

  HANDLE_ORDER_GEMMBATCH;
  FUNC_INIT;

  {
    size_t i;
    for (i = 0; i < batchCount; i++) {
      ARRAY_INIT(A[i]);
      A_l[i] = ((float *)A[i]->ptr) + offA[i];
      ARRAY_INIT(B[i]);
      B_l[i] = ((float *)B[i]->ptr) + offB[i];
      ARRAY_INIT(C[i]);
      C_l[i] = ((float *)C[i]->ptr) + offC[i];
    }
  }

  PRE_CALL cublasSgemmBatched(INIT_ARGS TRANS(transA), TRANS(transB), SZ(M), SZ(N), SZ(K), SCAL(alpha), A_l, SZ(lda), B_l, SZ(ldb), SCAL(beta), C_l, SZ(ldc), batchCount);
  POST_CALL;

  {
    size_t i;
    for (i = 0; i < batchCount; i++) {
      ARRAY_FINI(A[i]);
      ARRAY_FINI(B[i]);
      ARRAY_FINI(C[i]);
    }
  }

  FUNC_FINI;

  return GA_NO_ERROR;
}

static int dgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, double alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      double beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  FETCH_CONTEXT(A[0]);
  FUNC_DECLS;
  const double **A_l = alloca(sizeof(double *) * batchCount);
  const double **B_l = alloca(sizeof(double *) * batchCount);
  double **C_l = alloca(sizeof(double *) * batchCount);
  PREP_ORDER_GEMMBATCH;

  HANDLE_ORDER_GEMMBATCH;
  FUNC_INIT;

  {
    size_t i;
    for (i = 0; i < batchCount; i++) {
      ARRAY_INIT(A[i]);
      A_l[i] = ((double *)A[i]->ptr) + offA[i];
      ARRAY_INIT(B[i]);
      B_l[i] = ((double *)B[i]->ptr) + offB[i];
      ARRAY_INIT(C[i]);
      C_l[i] = ((double *)C[i]->ptr) + offC[i];
    }
  }

  PRE_CALL cublasDgemmBatched(INIT_ARGS TRANS(transA), TRANS(transB), SZ(M), SZ(N), SZ(K), SCAL(alpha), A_l, SZ(lda), B_l, SZ(ldb), SCAL(beta), C_l, SZ(ldc), batchCount);
  POST_CALL;

  {
    size_t i;
    for (i = 0; i < batchCount; i++) {
      ARRAY_FINI(A[i]);
      ARRAY_FINI(B[i]);
      ARRAY_FINI(C[i]);
    }
  }

  FUNC_FINI;

  return GA_NO_ERROR;
}

#include "generic_blas.inc.c"

