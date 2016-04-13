#include "private.h"
#include "private_cuda.h"

#include "gpuarray/buffer_blas.h"
#include "gpuarray/error.h"

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
  err = cublasCreate(&handle);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    return GA_BLAS_ERROR;
  }

  err = cublasSetStream(handle, ctx->s);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(handle);
    cuda_exit(ctx);
    return GA_BLAS_ERROR;
  }

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);

  cuda_exit(ctx);

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

static int sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb,
                 float beta, gpudata *C, size_t offC, size_t ldc) {
  cuda_context *ctx = A->ctx;
  gpudata *T;
  size_t t;
  cublasStatus_t err;
  cb_transpose transT;

  ASSERT_BUF(A);
  ASSERT_BUF(B);
  ASSERT_BUF(C);

  if (order == cb_c) {
    /* swap A and B */
    t = N;
    N = M;
    M = t;
    T = A;
    A = B;
    B = T;
    t = lda;
    lda = ldb;
    ldb = t;
    transT = transA;
    transA = transB;
    transB = transT;
    t = offA;
    offA = offB;
    offB = t;
  }

  cuda_enter(ctx);

  cuda_wait(A, CUDA_WAIT_READ);
  cuda_wait(B, CUDA_WAIT_READ);
  cuda_wait(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasSgemm(ctx->blas_handle, convT(transA), convT(transB), M, N, K,
                    &alpha, ((float *)A->ptr) + offA, lda,
                    ((float *)B->ptr) + offB, ldb, &beta,
                    ((float *)C->ptr) + offC, ldc);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(B, CUDA_WAIT_READ);
  cuda_record(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int dgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, double alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb,
                 double beta, gpudata *C, size_t offC, size_t ldc) {
  cuda_context *ctx = A->ctx;
  gpudata *T;
  size_t t;
  cublasStatus_t err;
  cb_transpose transT;

  ASSERT_BUF(A);
  ASSERT_BUF(B);
  ASSERT_BUF(C);

  if (order == cb_c) {
    /* swap A and B */
    t = N;
    N = M;
    M = t;
    T = A;
    A = B;
    B = T;
    t = lda;
    lda = ldb;
    ldb = t;
    transT = transA;
    transA = transB;
    transB = transT;
    t = offA;
    offA = offB;
    offB = t;
  }

  cuda_enter(ctx);

  cuda_wait(A, CUDA_WAIT_READ);
  cuda_wait(B, CUDA_WAIT_READ);
  cuda_wait(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasDgemm(ctx->blas_handle, convT(transA), convT(transB), M, N, K,
                    &alpha, ((double *)A->ptr) + offA, lda,
                    ((double *)B->ptr) + offB, ldb, &beta,
                    ((double *)C->ptr) + offC, ldc);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(B, CUDA_WAIT_READ);
  cuda_record(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int hgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb,
                 float beta, gpudata *C, size_t offC, size_t ldc) {
#ifdef HAVE_CUBLAS_SGEMMEX
  /* This will use float32 for computation as it's the best we can
   * have right now. In the future when native float16 support will be
   * there we will switch to that. */
  cuda_context *ctx = A->ctx;
  gpudata *T;
  size_t t;
  cublasStatus_t err;
  cb_transpose transT;

  ASSERT_BUF(A);
  ASSERT_BUF(B);
  ASSERT_BUF(C);

  if (order == cb_c) {
    /* swap A and B */
    t = N;
    N = M;
    M = t;
    T = A;
    A = B;
    B = T;
    t = lda;
    lda = ldb;
    ldb = t;
    transT = transA;
    transA = transB;
    transB = transT;
    t = offA;
    offA = offB;
    offB = t;
  }

  cuda_enter(ctx);

  cuda_wait(A, CUDA_WAIT_READ);
  cuda_wait(B, CUDA_WAIT_READ);
  cuda_wait(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasSgemmEx(ctx->blas_handle, convT(transA), convT(transB), M, N, K,
                      &alpha,
                      ((uint16_t *)A->ptr) + offA, CUBLAS_DATA_HALF, lda,
                      ((uint16_t *)B->ptr) + offB, CUBLAS_DATA_HALF, ldb,
                      &beta,
                      ((uint16_t *)C->ptr) + offC, CUBLAS_DATA_HALF, ldc);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(B, CUDA_WAIT_READ);
  cuda_record(C, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);
  return GA_NO_ERROR;
#else
  return GA_DEVSUP_ERROR;
#endif
}

static int sgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cuda_context *ctx;
  size_t *lt, t;
  gpudata **T;
  size_t i;
  cb_transpose transT;
  cublasStatus_t err;

  if (batchCount == 0) return GA_NO_ERROR;

  ASSERT_BUF(A[0]);
  ctx = A[0]->ctx;
  cuda_enter(ctx);

  if (order == cb_c) {
    /* swap A and B */
    t = N;
    N = M;
    M = t;
    T = A;
    A = B;
    B = T;
    t = lda;
    lda = ldb;
    ldb = t;
    transT = transA;
    transA = transB;
    transB = transT;
    lt = offA;
    offA = offB;
    offB = lt;
  }

  // use parallel cublasSgemm calls rather than cublasSgemmBatched for large products
  // (compute products in double because they can be large and we don't need to be exact)
  const double threshold = 650;
  const int multiple_dispatch = ((double)M * (double)N * (double)K >
                                 threshold * threshold * threshold);
  if (multiple_dispatch) {
    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(B[i]);
      ASSERT_BUF(C[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(B[i], CUDA_WAIT_READ);
      cuda_wait(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);

      err = cublasSgemm(ctx->blas_handle, convT(transA), convT(transB),
                        M, N, K, &alpha,
                        (float*)A[i]->ptr + offA[i], lda,
                        (float*)B[i]->ptr + offB[i], ldb,
                        &beta,
                        (float*)C[i]->ptr + offC[i], ldc);
      if (err != CUBLAS_STATUS_SUCCESS) {
        cuda_exit(ctx);
        if (err == CUBLAS_STATUS_ARCH_MISMATCH)
          return GA_DEVSUP_ERROR;
        return GA_BLAS_ERROR;
      }

      cuda_record(A[i], CUDA_WAIT_READ);
      cuda_record(B[i], CUDA_WAIT_READ);
      cuda_record(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
    }
  } else {
    float **T_l = alloca(sizeof(float *) * batchCount * 3);
    const float **A_l = (const float **)T_l;
    const float **B_l = (const float **)T_l + batchCount;
    float **C_l = T_l + (batchCount * 2);
    CUdeviceptr Ta, Aa, Ba, Ca;

    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(B[i]);
      ASSERT_BUF(C[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(B[i], CUDA_WAIT_READ);
      cuda_wait(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      A_l[i] = ((float *)A[i]->ptr) + offA[i];
      B_l[i] = ((float *)B[i]->ptr) + offB[i];
      C_l[i] = ((float *)C[i]->ptr) + offC[i];
    }

    cuMemAlloc(&Ta, sizeof(float *) * batchCount * 3);
    Aa = Ta;
    Ba = Ta + (batchCount * sizeof(float *));
    Ca = Ta + (batchCount * sizeof(float *) * 2);

    cuMemcpyHtoD(Ta, T_l, sizeof(float *) * batchCount * 3);

    err = cublasSgemmBatched(ctx->blas_handle, convT(transA), convT(transB),
                             M, N, K, &alpha, (const float **)Aa, lda,
                             (const float **)Ba, ldb, &beta,
                             (float **)Ca, ldc, batchCount);
    cuMemFree(Ta);
    if (err != CUBLAS_STATUS_SUCCESS) {
      cuda_exit(ctx);
      if (err == CUBLAS_STATUS_ARCH_MISMATCH)
        return GA_DEVSUP_ERROR;
      return GA_BLAS_ERROR;
    }

    for (i = 0; i < batchCount; i++) {
      cuda_record(A[i], CUDA_WAIT_READ);
      cuda_record(B[i], CUDA_WAIT_READ);
      cuda_record(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
    }
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int dgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, double alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      double beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cuda_context *ctx;
  size_t *lt, t;
  gpudata **T;
  size_t i;
  cb_transpose transT;
  cublasStatus_t err;

  if (batchCount == 0) return GA_NO_ERROR;

  ASSERT_BUF(A[0]);

  ctx = A[0]->ctx;

  /* Possibly optimize this to make multiple dispatch of sgemm for
   * bigger sizes */
  double **T_l = alloca(sizeof(double *) * batchCount * 3);
  const double **A_l = (const double **)T_l;
  const double **B_l = (const double **)T_l + batchCount;
  double **C_l = T_l + (batchCount * 2);
  CUdeviceptr Ta, Aa, Ba, Ca;

  if (order == cb_c) {
    /* swap A and B */
    t = N;
    N = M;
    M = t;
    T = A;
    A = B;
    B = T;
    t = lda;
    lda = ldb;
    ldb = t;
    transT = transA;
    transA = transB;
    transB = transT;
    lt = offA;
    offA = offB;
    offB = lt;
  }

  cuda_enter(ctx);

  for (i = 0; i < batchCount; i++) {
    ASSERT_BUF(A[i]);
    ASSERT_BUF(B[i]);
    ASSERT_BUF(C[i]);
    cuda_wait(A[i], CUDA_WAIT_READ);
    cuda_wait(B[i], CUDA_WAIT_READ);
    cuda_wait(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
    A_l[i] = ((double *)A[i]->ptr) + offA[i];
    B_l[i] = ((double *)B[i]->ptr) + offB[i];
    C_l[i] = ((double *)C[i]->ptr) + offC[i];
  }

  cuMemAlloc(&Ta, sizeof(double *) * batchCount * 3);
  Aa = Ta;
  Ba = Ta + (batchCount * sizeof(double *));
  Ca = Ta + (batchCount * sizeof(double *) * 2);

  cuMemcpyHtoD(Ta, T_l, sizeof(double *) * batchCount * 3);

  err = cublasDgemmBatched(ctx->blas_handle, convT(transA), convT(transB),
                           M, N, K, &alpha, (const double **)Aa, lda,
                           (const double **)Ba, ldb, &beta,
                           (double **)Ca, ldc, batchCount);
  cuMemFree(Ta);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  for (i = 0; i < batchCount; i++) {
    cuda_record(A[i], CUDA_WAIT_READ);
    cuda_record(B[i], CUDA_WAIT_READ);
    cuda_record(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int sgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX,
                 float beta, gpudata *Y, size_t offY, int incY) {
  cuda_context *ctx = A->ctx;
  cublasStatus_t err;
  size_t t;

  ASSERT_BUF(A);
  ASSERT_BUF(X);
  ASSERT_BUF(Y);

  if (order == cb_c) {
    t = N;
    N = M;
    M = t;

    if (transA == cb_no_trans) {
      transA = cb_trans;
    } else {
      transA = cb_no_trans;
    }
  }

  cuda_enter(ctx);

  cuda_wait(A, CUDA_WAIT_READ);
  cuda_wait(X, CUDA_WAIT_READ);
  cuda_wait(Y, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasSgemv(ctx->blas_handle, convT(transA), M, N, &alpha,
                    ((float *)A->ptr) + offA, lda,
                    ((float *)X->ptr) + offX, incX,
                    &beta, ((float *)Y->ptr) + offY, incY);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

static int dgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 double alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX,
                 double beta, gpudata *Y, size_t offY, int incY) {
  cuda_context *ctx = A->ctx;
  cublasStatus_t err;
  size_t t;

  ASSERT_BUF(A);
  ASSERT_BUF(X);
  ASSERT_BUF(Y);

  if (order == cb_c) {
    t = N;
    N = M;
    M = t;

    if (transA == cb_no_trans) {
      transA = cb_trans;
    } else {
      transA = cb_no_trans;
    }
  }

  cuda_enter(ctx);

  cuda_wait(A, CUDA_WAIT_READ);
  cuda_wait(X, CUDA_WAIT_READ);
  cuda_wait(Y, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasDgemv(ctx->blas_handle, convT(transA), M, N, &alpha,
                    ((double *)A->ptr) + offA, lda,
                    ((double *)X->ptr) + offX, incX,
                    &beta, ((double *)Y->ptr) + offY, incY);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

static int sger(cb_order order, size_t M, size_t N, float alpha, gpudata *X,
                size_t offX, int incX, gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cuda_context *ctx = X->ctx;
  gpudata *td;
  size_t t;
  cublasStatus_t err;

  ASSERT_BUF(X);
  ASSERT_BUF(Y);
  ASSERT_BUF(A);

  if (order == cb_c) {
    t = M;
    M = N;
    N = t;
    t = offX;
    offX = offY;
    offY = t;
    t = incX;
    incX = incY;
    incY = t;
    td = X;
    X = Y;
    Y = td;
  }

  cuda_enter(ctx);

  cuda_wait(X, CUDA_WAIT_READ);
  cuda_wait(Y, CUDA_WAIT_READ);
  cuda_wait(A, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasSger(ctx->blas_handle, M, N, &alpha,
                   ((float *)X->ptr) + offX, incX,
                   ((float *)Y->ptr) + offY, incY,
                   ((float *)A->ptr) + offA, lda);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ);
  cuda_record(A, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

static int dger(cb_order order, size_t M, size_t N, double alpha, gpudata *X,
                size_t offX, int incX, gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cuda_context *ctx = X->ctx;
  gpudata *td;
  size_t t;
  cublasStatus_t err;

  ASSERT_BUF(X);
  ASSERT_BUF(Y);
  ASSERT_BUF(A);

  if (order == cb_c) {
    t = M;
    M = N;
    N = t;
    t = offX;
    offX = offY;
    offY = t;
    t = incX;
    incX = incY;
    incY = t;
    td = X;
    X = Y;
    Y = td;
  }

  cuda_enter(ctx);

  cuda_wait(X, CUDA_WAIT_READ);
  cuda_wait(Y, CUDA_WAIT_READ);
  cuda_wait(A, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  err = cublasDger(ctx->blas_handle, M, N, &alpha,
                   ((double *)X->ptr) + offX, incX,
                   ((double *)Y->ptr) + offY, incY,
                   ((double *)A->ptr) + offA, lda);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ);
  cuda_record(A, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

GPUARRAY_LOCAL gpuarray_blas_ops cublas_ops = {
  setup,
  teardown,
  NULL, /* hgemv */
  sgemv,
  dgemv,
  hgemm,
  sgemm,
  dgemm,
  NULL, /* hger */
  sger,
  dger,
  NULL, /* hgemmBatch */
  sgemmBatch,
  dgemmBatch,
};
