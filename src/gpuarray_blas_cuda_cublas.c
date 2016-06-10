#include "private.h"
#include "private_cuda.h"

#include "gpuarray/buffer_blas.h"
#include "gpuarray/kernel.h"
#include "gpuarray/error.h"

#include "cublas_v2.h"

extern const gpuarray_buffer_ops cuda_ops;

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

typedef struct _blas_handle {
  cublasHandle_t h;
  GpuKernel sgemvBH_N_a1_b1_small;
  GpuKernel sgemvBH_T_a1_b1_small;
  GpuKernel dgemvBH_N_a1_b1_small;
  GpuKernel dgemvBH_T_a1_b1_small;
  GpuKernel sgerBH_gen_small;
  GpuKernel dgerBH_gen_small;
  cublasStatus_t err;
} blas_handle;

static const char *code_sgemvBH_N_a1_b1_small =                         \
  "extern \"C\"__global__ void sgemv(const float *A[], size_t lda, "    \
  "                                  const float *x[], size_t incx, "   \
  "                                  float *y[], size_t incy, "         \
  "                                  size_t b, size_t m, size_t n) {"   \
  "  for (size_t p = blockIdx.y * blockDim.y + threadIdx.y; p < b;"     \
  "       p += gridDim.y * blockDim.y) {"                               \
  "    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;"   \
  "         i += gridDim.x * blockDim.x) {"                             \
  "      float yi = 0.0f;"                                              \
  "      const float *Ap = A[p] + i;"                                   \
  "      const float *xp = x[p];\n"                                     \
  "      #pragma unroll 32\n"                                           \
  "      for (size_t j = 0; j < n; j++) {"                              \
  "        yi += Ap[0] * xp[0];"                                        \
  "        Ap += lda;"                                                  \
  "        xp += incx;"                                                 \
  "      }"                                                             \
  "     atomicAdd(&y[p][i*incy], yi);"                                  \
  "    }"                                                               \
  "  }"                                                                 \
  "}\n";

static const char *code_sgemvBH_T_a1_b1_small =                         \
  "extern \"C\" __global__ void sgemv(const float *A[], size_t lda, "   \
  "                                   const float *x[], size_t incx, "  \
  "                                   float *y[], size_t incy, "        \
  "                                   size_t b, size_t m, size_t n) {"  \
  "  size_t i = blockIdx.x * blockDim.x + threadIdx.x;"                 \
  "  size_t p = blockIdx.y * blockDim.y + threadIdx.y;"                 \
  "  if (i >= m || p >= b) return;"                                     \
  "  float yi = 0.0f;"                                                  \
  "  const float *Ap = A[p] + i * lda;"                                 \
  "  const float *xp = x[p];\n"                                         \
  "  # pragma unroll 32\n"                                              \
  "  for (size_t j = 0; j < n; j++) {"                                  \
  "    yi += Ap[j] * xp[0];"                                            \
  "    xp += incx;"                                                     \
  "  }"                                                                 \
  "  atomicAdd(&y[p][i*incy], yi);"                                     \
  "}\n";

static const char *atomicadd_double =                                   \
  "__device__ double atomicAdd(double* address, double val) {"          \
  "  unsigned long long int* address_as_ull ="                          \
  "  (unsigned long long int*)address;"                                 \
  "  unsigned long long int old = *address_as_ull, assumed;"            \
  "  do {"                                                              \
  "    assumed = old;"                                                  \
  "    old = atomicCAS(address_as_ull, assumed,"                        \
  "                    __double_as_longlong(val +"                      \
  "                    __longlong_as_double(assumed)));"                \
  "  } while (assumed != old);"                                         \
  "  return __longlong_as_double(old);"                                 \
  "}\n";

static const char *code_dgemvBH_N_a1_b1_small =                         \
  "extern \"C\" __global__ void dgemv(const double *A[], size_t lda, "  \
  "                                   const double *x[], size_t incx, " \
  "                                   double *y[], size_t incy, "       \
  "                                   size_t b, size_t m, size_t n) {"  \
  "  for (size_t p = blockIdx.y * blockDim.y + threadIdx.y; p < b;"     \
  "       p += gridDim.y * blockDim.y) {"                               \
  "    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;"   \
  "         i += gridDim.x * blockDim.x) {"                             \
  "      double yi = 0.0;"                                              \
  "      const double *Ap = A[p] + i;"                                  \
  "      const double *xp = x[p];\n"                                    \
  "      #pragma unroll 32\n"                                           \
  "      for (size_t j = 0; j < n; j++) {"                              \
  "        yi += Ap[0] * xp[0];"                                        \
  "        Ap += lda;"                                                  \
  "        xp += incx;"                                                 \
  "      }"                                                             \
  "     atomicAdd(&y[p][i*incy], yi);"                                  \
  "    }"                                                               \
  "  }"                                                                 \
  "}\n";

static const char *code_dgemvBH_T_a1_b1_small =                         \
  "extern \"C\" __global__ void dgemv(const double *A[], size_t lda, "  \
  "                                   const double *x[], size_t incx, " \
  "                                   double *y[], size_t incy, "       \
  "                                   size_t b, size_t m, size_t n) {"  \
  "  size_t i = blockIdx.x * blockDim.x + threadIdx.x;"                 \
  "  size_t p = blockIdx.y * blockDim.y + threadIdx.y;"                 \
  "  if (i >= m || p >= b) return;"                                     \
  "  double yi = 0.0;"                                                  \
  "  const double *Ap = A[p] + i * lda;"                                \
  "  const double *xp = x[p];\n"                                        \
  "  # pragma unroll 32\n"                                              \
  "  for (size_t j = 0; j < n; j++) {"                                  \
  "    yi += Ap[j] * xp[0];"                                            \
  "    xp += incx;"                                                     \
  "  }"                                                                 \
  "  atomicAdd(&y[p][i*incy], yi);"                                     \
  "}\n";

static const char *code_sgerBH_gen_small =                              \
  "extern \"C\" __global__ void _sgerBH_gen_small("                     \
  "    const float *x[], size_t incx,"                                  \
  "    const float *y[], size_t incy,"                                  \
  "    float alpha, float *A[], size_t lda,"                            \
  "    size_t b, size_t m, size_t n) {"                                 \
  "  size_t i = blockIdx.x * blockDim.x + threadIdx.x;"                 \
  "  size_t j = blockIdx.y * blockDim.y + threadIdx.y;"                 \
  "  if (i >= m || j >= n) return;"                                     \
  "  for (size_t p = blockIdx.z; p < b; p += gridDim.z) {"              \
  "    atomicAdd(&A[p][j * lda + i],"                                   \
  "              alpha * x[p][i * incx] * y[p][j * incy]);"             \
  "  }"                                                                 \
  "}\n";

static const char *code_dgerBH_gen_small =                              \
  "extern \"C\" __global__ void _dgerBH_gen_small("                     \
  "      const double *x[], size_t incx, "                              \
  "      const double *y[], size_t incy,"                               \
  "      double alpha, double *A[], size_t lda,"                        \
  "      size_t b, size_t m, size_t n) {"                               \
  "  size_t i = blockIdx.x * blockDim.x + threadIdx.x;"                 \
  "  size_t j = blockIdx.y * blockDim.y + threadIdx.y;"                 \
  "  if (i >= m || j >= n) return;"                                     \
  "  for (size_t p = blockIdx.z; p < b; p += gridDim.z) {"              \
  "    atomicAdd(&A[p][j * lda + i],"                                   \
  "              alpha * x[p][i * incx] * y[p][j * incy]);"             \
  "  }"                                                                 \
  "}\n";

static int setup(gpucontext *c) {
  cuda_context *ctx = (cuda_context *)c;
  blas_handle *handle;
  const char *tmp[2];
  cublasStatus_t err;
  int e;
  int types[10];

  if (ctx->blas_handle != NULL)
    return GA_NO_ERROR;

  handle = calloc(1, sizeof(*handle));
  if (handle == NULL)
    return GA_MEMORY_ERROR;

  handle->err = CUBLAS_STATUS_SUCCESS;

  cuda_enter(ctx);
  err = cublasCreate(&handle->h);
  if (err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    free(handle);
    return GA_BLAS_ERROR;
  }

  err = cublasSetStream(handle->h, ctx->s);
  if (err != CUBLAS_STATUS_SUCCESS) {
    e = GA_BLAS_ERROR;
    goto e1;
  }

  cublasSetPointerMode(handle->h, CUBLAS_POINTER_MODE_HOST);
  cublasSetAtomicsMode(handle->h, CUBLAS_ATOMICS_ALLOWED);

  types[0] = GA_BUFFER;
  types[1] = GA_SIZE;
  types[2] = GA_BUFFER;
  types[3] = GA_SIZE;
  types[4] = GA_BUFFER;
  types[5] = GA_SIZE;
  types[6] = GA_SIZE;
  types[7] = GA_SIZE;
  types[8] = GA_SIZE;
  e = GpuKernel_init(&handle->sgemvBH_N_a1_b1_small, c, 1, &code_sgemvBH_N_a1_b1_small, NULL, "sgemv", 9, types, 0, NULL);
  if (e != GA_NO_ERROR) goto e1;
  e = GpuKernel_init(&handle->sgemvBH_T_a1_b1_small, c, 1, &code_sgemvBH_T_a1_b1_small, NULL, "sgemv", 9, types, 0, NULL);
  if (e != GA_NO_ERROR) goto e2;
  tmp[0] = atomicadd_double;
  tmp[1] = code_dgemvBH_N_a1_b1_small;
  e = GpuKernel_init(&handle->dgemvBH_N_a1_b1_small, c, 2, tmp, NULL, "dgemv", 9, types, GA_USE_DOUBLE, NULL);
  if (e != GA_NO_ERROR) goto e3;
  tmp[0] = atomicadd_double;
  tmp[1] = code_dgemvBH_T_a1_b1_small;
  e = GpuKernel_init(&handle->dgemvBH_T_a1_b1_small, c, 2, tmp, NULL, "dgemv", 9, types, GA_USE_DOUBLE, NULL);
  if (e != GA_NO_ERROR) goto e4;

  types[0] = GA_BUFFER;
  types[1] = GA_SIZE;
  types[2] = GA_BUFFER;
  types[3] = GA_SIZE;
  types[4] = GA_FLOAT;
  types[5] = GA_BUFFER;
  types[6] = GA_SIZE;
  types[7] = GA_SIZE;
  types[8] = GA_SIZE;
  types[9] = GA_SIZE;
  e = GpuKernel_init(&handle->sgerBH_gen_small, c, 1, &code_sgerBH_gen_small, NULL, "_sgerBH_gen_small", 10, types, 0, NULL);
  if (e != GA_NO_ERROR) goto e5;
  types[4] = GA_DOUBLE;
  tmp[0] = atomicadd_double;
  tmp[1] = code_dgerBH_gen_small;
  e = GpuKernel_init(&handle->dgerBH_gen_small, c, 2, tmp, NULL, "_dgerBH_gen_small", 10, types, GA_USE_DOUBLE, NULL);
  if (e != GA_NO_ERROR) goto e6;

  ctx->blas_handle = handle;

  cuda_exit(ctx);

  return GA_NO_ERROR;

 e6:
  GpuKernel_clear(&handle->sgerBH_gen_small);
 e5:
  GpuKernel_clear(&handle->dgemvBH_T_a1_b1_small);
 e4:
  GpuKernel_clear(&handle->dgemvBH_N_a1_b1_small);
 e3:
  GpuKernel_clear(&handle->sgemvBH_T_a1_b1_small);
 e2:
  GpuKernel_clear(&handle->sgemvBH_N_a1_b1_small);
 e1:
  cublasDestroy(handle->h);
  cuda_exit(ctx);
  free(handle);
  return e;
}

static void teardown(gpucontext *c) {
  cuda_context *ctx = (cuda_context *)c;
  blas_handle *handle = (blas_handle *)ctx->blas_handle;

  if (ctx->blas_handle == NULL)
    return;

  cuda_enter(ctx);
  cublasDestroy(handle->h);
  GpuKernel_clear(&handle->sgemvBH_N_a1_b1_small);
  GpuKernel_clear(&handle->sgemvBH_T_a1_b1_small);
  GpuKernel_clear(&handle->dgemvBH_N_a1_b1_small);
  GpuKernel_clear(&handle->dgemvBH_T_a1_b1_small);
  GpuKernel_clear(&handle->sgerBH_gen_small);
  GpuKernel_clear(&handle->dgerBH_gen_small);
  cuda_exit(ctx);
  free(ctx->blas_handle);
  ctx->blas_handle = NULL;
}

static const char *error(gpucontext *c) {
  cuda_context *ctx = (cuda_context *)c;
  blas_handle *handle = (blas_handle *)ctx->blas_handle;

  if (handle != NULL) {
    switch (handle->err) {
    case CUBLAS_STATUS_SUCCESS:
      return "(cublas) Operation completed successfully.";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "(cublas) Library not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "(cublas) GPU ressource allocation failed.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "(cublas) Invalid value.";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "(cublas) Operation not supported by device.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "(cublas) Mapping error.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "(cublas) Execution failed.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "(cublas) Internal error.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "(cublas) Unsupported functionality.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "(cublas) License error.";
    default:
      return "(cublas) Unknown error.";
    }
  }
  return "Blas handle not initialized, API error.";
}

static int sgemm(cb_order order, cb_transpose transA, cb_transpose transB,
                 size_t M, size_t N, size_t K, float alpha,
                 gpudata *A, size_t offA, size_t lda,
                 gpudata *B, size_t offB, size_t ldb,
                 float beta, gpudata *C, size_t offC, size_t ldc) {
  cuda_context *ctx = A->ctx;
  blas_handle *h = (blas_handle *)ctx->blas_handle;
  gpudata *T;
  size_t t;
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

  h->err = cublasSgemm(h->h,
                       convT(transA), convT(transB), M, N, K,
                       &alpha, ((float *)A->ptr) + offA, lda,
                       ((float *)B->ptr) + offB, ldb, &beta,
                       ((float *)C->ptr) + offC, ldc);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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
  blas_handle *h = (blas_handle *)ctx->blas_handle;
  gpudata *T;
  size_t t;
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

  h->err = cublasDgemm(h->h,
                       convT(transA), convT(transB), M, N, K,
                       &alpha, ((double *)A->ptr) + offA, lda,
                       ((double *)B->ptr) + offB, ldb, &beta,
                       ((double *)C->ptr) + offC, ldc);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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
  blas_handle *h = (blas_handle *)ctx->blas_handle;
  gpudata *T;
  size_t t;
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

  h->err = cublasSgemmEx(h->h,
                         convT(transA), convT(transB), M, N, K,
                         &alpha, ((uint16_t *)A->ptr) + offA,
#if CUDA_VERSION >= 8000
                         CUDA_R_16F,
#else
                         CUBLAS_DATA_HALF,
#endif
                         lda, ((uint16_t *)B->ptr) + offB,
#if CUDA_VERSION >= 8000
                         CUDA_R_16F,
#else
                         CUBLAS_DATA_HALF,
#endif
                         ldb, &beta, ((uint16_t *)C->ptr) + offC,
#if CUDA_VERSION >= 8000
                         CUDA_R_16F,
#else
                         CUBLAS_DATA_HALF,
#endif
                         ldc);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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

static int hgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  return GA_DEVSUP_ERROR;
}

static int sgemmBatch(cb_order order, cb_transpose transA, cb_transpose transB,
                      size_t M, size_t N, size_t K, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **B, size_t *offB, size_t ldb,
                      float beta, gpudata **C, size_t *offC, size_t ldc,
                      size_t batchCount) {
  cuda_context *ctx;
  blas_handle *h;
  size_t *lt, t;
  gpudata **T;
  size_t i;
  cb_transpose transT;

  if (batchCount == 0) return GA_NO_ERROR;

  ASSERT_BUF(A[0]);
  ctx = A[0]->ctx;
  h = (blas_handle *)ctx->blas_handle;
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
  const size_t threshold = 650;
  const int multiple_dispatch = M * N * K > threshold * threshold * threshold;
  if (multiple_dispatch) {
    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(B[i]);
      ASSERT_BUF(C[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(B[i], CUDA_WAIT_READ);
      cuda_wait(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);

      h->err = cublasSgemm(h->h,
                           convT(transA), convT(transB),
                           M, N, K, &alpha,
                           (float*)A[i]->ptr + offA[i], lda,
                           (float*)B[i]->ptr + offB[i], ldb,
                           &beta,
                           (float*)C[i]->ptr + offC[i], ldc);
      if (h->err != CUBLAS_STATUS_SUCCESS) {
        cuda_exit(ctx);
        if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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

    h->err = cublasSgemmBatched(h->h,
                                convT(transA), convT(transB),
                                M, N, K, &alpha,
                                (const float **)Aa, lda,
                                (const float **)Ba, ldb, &beta,
                                (float **)Ca, ldc, batchCount);
    cuMemFree(Ta);
    if (h->err != CUBLAS_STATUS_SUCCESS) {
      cuda_exit(ctx);
      if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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
  blas_handle *h;
  size_t *lt, t;
  gpudata **T;
  size_t i;
  cb_transpose transT;

  if (batchCount == 0) return GA_NO_ERROR;

  ASSERT_BUF(A[0]);
  ctx = A[0]->ctx;
  h = (blas_handle *)ctx->blas_handle;
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
  const size_t threshold = 650;
  const int multiple_dispatch = M * N * K > threshold * threshold * threshold;
  if (multiple_dispatch) {
    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(B[i]);
      ASSERT_BUF(C[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(B[i], CUDA_WAIT_READ);
      cuda_wait(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);

      h->err = cublasDgemm(h->h,
                           convT(transA), convT(transB),
                           M, N, K, &alpha,
                           (double*)A[i]->ptr + offA[i], lda,
                           (double*)B[i]->ptr + offB[i], ldb,
                           &beta,
                           (double*)C[i]->ptr + offC[i], ldc);
      if (h->err != CUBLAS_STATUS_SUCCESS) {
        cuda_exit(ctx);
        if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
          return GA_DEVSUP_ERROR;
        return GA_BLAS_ERROR;
      }

      cuda_record(A[i], CUDA_WAIT_READ);
      cuda_record(B[i], CUDA_WAIT_READ);
      cuda_record(C[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
    }
  } else {
    double **T_l = alloca(sizeof(double *) * batchCount * 3);
    const double **A_l = (const double **)T_l;
    const double **B_l = (const double **)T_l + batchCount;
    double **C_l = T_l + (batchCount * 2);
    CUdeviceptr Ta, Aa, Ba, Ca;

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

    h->err = cublasDgemmBatched(h->h,
                                convT(transA), convT(transB),
                                M, N, K, &alpha,
                                (const double **)Aa, lda,
                                (const double **)Ba, ldb, &beta,
                                (double **)Ca, ldc, batchCount);
    cuMemFree(Ta);
    if (h->err != CUBLAS_STATUS_SUCCESS) {
      cuda_exit(ctx);
      if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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

static int hgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX,
                 float beta, gpudata *Y, size_t offY, int incY) {
  return GA_DEVSUP_ERROR;
}

static int sgemv(cb_order order, cb_transpose transA, size_t M, size_t N,
                 float alpha, gpudata *A, size_t offA, size_t lda,
                 gpudata *X, size_t offX, int incX,
                 float beta, gpudata *Y, size_t offY, int incY) {
  cuda_context *ctx = A->ctx;
  blas_handle *h = (blas_handle *)ctx->blas_handle;
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

  h->err = cublasSgemv(h->h,
                       convT(transA), M, N, &alpha,
                       ((float *)A->ptr) + offA, lda,
                       ((float *)X->ptr) + offX, incX,
                       &beta, ((float *)Y->ptr) + offY, incY);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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
  blas_handle *h = (blas_handle *)ctx->blas_handle;
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

  h->err = cublasDgemv(h->h,
                       convT(transA), M, N, &alpha,
                       ((double *)A->ptr) + offA, lda,
                       ((double *)X->ptr) + offX, incX,
                       &beta, ((double *)Y->ptr) + offY, incY);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(A, CUDA_WAIT_READ);
  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

static int hgemvBatch(cb_order order, cb_transpose transA,
                      size_t M, size_t N, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **x, size_t *offX, size_t incX,
                      float beta, gpudata **y, size_t *offY, size_t incY,
                      size_t batchCount, int flags) {
  return GA_DEVSUP_ERROR;
}

static int sgemvBatch(cb_order order, cb_transpose transA,
                      size_t M, size_t N, float alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **x, size_t *offX, size_t incX,
                      float beta, gpudata **y, size_t *offY, size_t incY,
                      size_t batchCount, int flags) {
  /* Flags is there for possible future implementations where we might
     not use atomics or have some alternate implemntation. */
  cuda_context *ctx;
  size_t t, i;
  size_t ls[2], gs[2];
  void *args[9];
  gpudata *Aa, *xa, *ya;
  int err;

  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;

  if (alpha != 1.0 || beta != 1.0) return GA_UNSUPPORTED_ERROR;

  if (M < 512) {
    ls[0] = 32;
    if (batchCount > 16)
      ls[1] = 16;
    else
      ls[1] = batchCount;
  } else {
    ls[0] = 512;
    ls[1] = 1;
  }
  gs[0] = (M + ls[0] - 1) / ls[0];
  gs[1] = (batchCount + ls[1] - 1) / ls[1];
  if (gs[0] * gs[1] / 65535) {
    gs[1] = (65535 / gs[0]);
  }

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

  ASSERT_BUF(A[0]);

  ctx = A[0]->ctx;

  cuda_enter(ctx);

  {
    float **T_l = alloca(sizeof(float *) * batchCount * 3);
    const float **A_l = (const float **)T_l;
    const float **x_l = (const float **)T_l + batchCount;
    float **y_l = T_l + (batchCount * 2);

    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(x[i]);
      ASSERT_BUF(y[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(x[i], CUDA_WAIT_READ);
      cuda_wait(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      A_l[i] = (float *)(A[i]->ptr + offA[i]);
      x_l[i] = (float *)(x[i]->ptr + offX[i]);
      y_l[i] = (float *)(y[i]->ptr + offY[i]);
    }

    Aa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, A_l,
                               GA_BUFFER_INIT, &err);
    if (Aa == NULL)
      return err;
    xa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, x_l,
                               GA_BUFFER_INIT, &err);
    if (xa == NULL) {
      cuda_ops.buffer_release(Aa);
      return err;
    }
    ya = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, y_l,
                               GA_BUFFER_INIT, &err);
    if (ya == NULL) {
      cuda_ops.buffer_release(Aa);
      cuda_ops.buffer_release(xa);
      return err;
    }
  }

  args[0] = Aa;
  args[1] = &lda;
  args[2] = xa;
  args[3] = &incX;
  args[4] = ya;
  args[5] = &incY;
  args[6] = &batchCount;
  args[7] = &M;
  args[8] = &N;

  if (transA == cb_no_trans) {
    err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->sgemvBH_N_a1_b1_small, 2, ls, gs, 0, args);
  } else {
    err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->sgemvBH_T_a1_b1_small, 2, ls, gs, 0, args);
  }

  cuda_ops.buffer_release(Aa);
  cuda_ops.buffer_release(xa);
  cuda_ops.buffer_release(ya);

  if (err != GA_NO_ERROR) {
    cuda_exit(ctx);
    return err;
  }


  for (i = 0; i < batchCount; i++) {
    cuda_record(A[i], CUDA_WAIT_READ);
    cuda_record(x[i], CUDA_WAIT_READ);
    cuda_record(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int dgemvBatch(cb_order order, cb_transpose transA,
                      size_t M, size_t N, double alpha,
                      gpudata **A, size_t *offA, size_t lda,
                      gpudata **x, size_t *offX, size_t incX,
                      double beta, gpudata **y, size_t *offY, size_t incY,
                      size_t batchCount, int flags) {
  cuda_context *ctx;
  size_t t, i;
  size_t ls[2], gs[2];
  void *args[9];
  gpudata *Aa, *xa, *ya;
  int err;

  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;

  if (alpha != 1.0 || beta != 1.0) return GA_UNSUPPORTED_ERROR;

  if (M < 512) {
    ls[0] = 32;
    if (batchCount > 16)
      ls[1] = 16;
    else
      ls[1] = batchCount;
  } else {
    ls[0] = 512;
    ls[1] = 1;
  }
  gs[0] = (M + ls[0] - 1) / ls[0];
  gs[1] = (batchCount + ls[1] - 1) / ls[1];
  if (gs[0] * gs[1] / 65535) {
    gs[1] = (65535 / gs[0]);
  }

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

  ASSERT_BUF(A[0]);

  ctx = A[0]->ctx;

  cuda_enter(ctx);

  {
    double **T_l = alloca(sizeof(double *) * batchCount * 3);
    const double **A_l = (const double **)T_l;
    const double **x_l = (const double **)T_l + batchCount;
    double **y_l = T_l + (batchCount * 2);

    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(x[i]);
      ASSERT_BUF(y[i]);
      cuda_wait(A[i], CUDA_WAIT_READ);
      cuda_wait(x[i], CUDA_WAIT_READ);
      cuda_wait(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      A_l[i] = (double *)(A[i]->ptr + offA[i]);
      x_l[i] = (double *)(x[i]->ptr + offX[i]);
      y_l[i] = (double *)(y[i]->ptr + offY[i]);
    }

    Aa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, A_l,
                               GA_BUFFER_INIT, &err);
    if (Aa == NULL)
      return err;
    xa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, x_l,
                               GA_BUFFER_INIT, &err);
    if (xa == NULL) {
      cuda_ops.buffer_release(Aa);
      return err;
    }
    ya = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, y_l,
                               GA_BUFFER_INIT, &err);
    if (ya == NULL) {
      cuda_ops.buffer_release(Aa);
      cuda_ops.buffer_release(xa);
      return err;
    }
  }

  args[0] = Aa;
  args[1] = &lda;
  args[2] = xa;
  args[3] = &incX;
  args[4] = ya;
  args[5] = &incY;
  args[6] = &batchCount;
  args[7] = &M;
  args[8] = &N;

  if (transA == cb_no_trans) {
    err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->dgemvBH_N_a1_b1_small, 2, ls, gs, 0, args);
  } else {
    err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->dgemvBH_T_a1_b1_small, 2, ls, gs, 0, args);
  }

  cuda_ops.buffer_release(Aa);
  cuda_ops.buffer_release(xa);
  cuda_ops.buffer_release(ya);

  if (err != GA_NO_ERROR) {
    cuda_exit(ctx);
    return err;
  }

  for (i = 0; i < batchCount; i++) {
    cuda_record(A[i], CUDA_WAIT_READ);
    cuda_record(x[i], CUDA_WAIT_READ);
    cuda_record(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}


static int hger(cb_order order, size_t M, size_t N, float alpha, gpudata *X,
                size_t offX, int incX, gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  return GA_DEVSUP_ERROR;
}

static int sger(cb_order order, size_t M, size_t N, float alpha, gpudata *X,
                size_t offX, int incX, gpudata *Y, size_t offY, int incY,
                gpudata *A, size_t offA, size_t lda) {
  cuda_context *ctx = X->ctx;
  blas_handle *h = (blas_handle *)ctx->blas_handle;
  gpudata *td;
  size_t t;

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

  h->err = cublasSger(h->h, M, N, &alpha,
                                ((float *)X->ptr) + offX, incX,
                                ((float *)Y->ptr) + offY, incY,
                                ((float *)A->ptr) + offA, lda);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
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
  blas_handle *h = (blas_handle *)ctx->blas_handle;
  gpudata *td;
  size_t t;

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

  h->err = cublasDger(h->h, M, N, &alpha,
                                ((double *)X->ptr) + offX, incX,
                                ((double *)Y->ptr) + offY, incY,
                                ((double *)A->ptr) + offA, lda);
  if (h->err != CUBLAS_STATUS_SUCCESS) {
    cuda_exit(ctx);
    if (h->err == CUBLAS_STATUS_ARCH_MISMATCH)
      return GA_DEVSUP_ERROR;
    return GA_BLAS_ERROR;
  }

  cuda_record(X, CUDA_WAIT_READ);
  cuda_record(Y, CUDA_WAIT_READ);
  cuda_record(A, CUDA_WAIT_READ|CUDA_WAIT_WRITE);

  cuda_exit(ctx);

  return GA_NO_ERROR;
}

static int hgerBatch(cb_order order, size_t M, size_t N, float alpha,
                     gpudata **x, size_t *offX, size_t incX,
                     gpudata **y, size_t *offY, size_t incY,
                     gpudata **A, size_t *offA, size_t lda,
                     size_t batchCount, int flags) {
  return GA_DEVSUP_ERROR;
}

static int sgerBatch(cb_order order, size_t M, size_t N, float alpha,
                     gpudata **x, size_t *offX, size_t incX,
                     gpudata **y, size_t *offY, size_t incY,
                     gpudata **A, size_t *offA, size_t lda,
                     size_t batchCount, int flags) {
  cuda_context *ctx;
  size_t t, *tp, i;
  size_t ls[3] = {M, N, 1}, gs[3] = {1, 1, batchCount};
  void *args[10];
  gpudata **T;
  gpudata *Aa, *xa, *ya;
  int err;

  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;

  if (incX == 1) {
    if (ls[0] > 32) {
      gs[0] = (ls[0] + 31) / 32;
      ls[0] = 32;
    }
    if (ls[0] * ls[1] > 512) {
      gs[1] = (ls[1] + 15) / 16;
      ls[1] = 16;
    }
  } else {
    if (ls[1] > 32) {
      gs[1] = (ls[1] + 31) / 32;
      ls[1] = 32;
    }
    if (ls[0] * ls[1] > 512) {
      gs[0] = (ls[0] + 15) / 16;
      ls[0] = 16;
    }
  }
  if (gs[0] * gs[1] * gs[2] > 65535) {
    if (gs[0] * gs[1] > 65535)
      return GA_VALUE_ERROR;
    gs[2] = (65535 / (gs[0] * gs[1]));
  }

  if (order == cb_c) {
    t = M;
    M = N;
    N = t;
    tp = offX;
    offX = offY;
    offY = tp;
    t = incX;
    incX = incY;
    incY = t;
    T = x;
    x = y;
    y = T;
  }

  ASSERT_BUF(x[0]);

  ctx = x[0]->ctx;

  cuda_enter(ctx);

  {
    float **T_l = alloca(sizeof(float *) * batchCount * 3);
    const float **A_l = (const float **)T_l;
    const float **x_l = (const float **)T_l + batchCount;
    float **y_l = T_l + (batchCount * 2);

    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(x[i]);
      ASSERT_BUF(y[i]);
      cuda_wait(A[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      cuda_wait(x[i], CUDA_WAIT_READ);
      cuda_wait(y[i], CUDA_WAIT_READ);
      A_l[i] = (float *)(A[i]->ptr + offA[i]);
      x_l[i] = (float *)(x[i]->ptr + offX[i]);
      y_l[i] = (float *)(y[i]->ptr + offY[i]);
    }

    Aa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, A_l,
                               GA_BUFFER_INIT, &err);
    if (Aa == NULL)
      return err;
    xa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, x_l,
                               GA_BUFFER_INIT, &err);
    if (xa == NULL) {
      cuda_ops.buffer_release(Aa);
      return err;
    }
    ya = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(float *) * batchCount, y_l,
                               GA_BUFFER_INIT, &err);
    if (ya == NULL) {
      cuda_ops.buffer_release(Aa);
      cuda_ops.buffer_release(xa);
      return err;
    }
  }

  args[0] = xa;
  args[1] = &incX;
  args[2] = ya;
  args[3] = &incY;
  args[4] = &alpha;
  args[5] = Aa;
  args[6] = &lda;
  args[7] = &batchCount;
  args[8] = &M;
  args[9] = &N;

  err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->sgerBH_gen_small, 3, ls, gs, 0, args);

  cuda_ops.buffer_release(Aa);
  cuda_ops.buffer_release(xa);
  cuda_ops.buffer_release(ya);

  if (err != GA_NO_ERROR) {
    cuda_exit(ctx);
    return err;
  }


  for (i = 0; i < batchCount; i++) {
    cuda_record(A[i], CUDA_WAIT_READ);
    cuda_record(x[i], CUDA_WAIT_READ);
    cuda_record(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

static int dgerBatch(cb_order order, size_t M, size_t N, double alpha,
                     gpudata **x, size_t *offX, size_t incX,
                     gpudata **y, size_t *offY, size_t incY,
                     gpudata **A, size_t *offA, size_t lda,
                     size_t batchCount, int flags) {
  cuda_context *ctx;
  size_t t, *tp, i;
  size_t ls[3] = {M, N, 1}, gs[3] = {1, 1, batchCount};
  void *args[10];
  gpudata **T;
  gpudata *Aa, *xa, *ya;
  int err;

  if (flags != 0) return GA_INVALID_ERROR;
  if (batchCount == 0) return GA_NO_ERROR;

  if (incX == 1) {
    if (ls[0] > 32) {
      gs[0] = (ls[0] + 31) / 32;
      ls[0] = 32;
    }
    if (ls[0] * ls[1] > 512) {
      gs[1] = (ls[1] + 15) / 16;
      ls[1] = 16;
    }
  } else {
    if (ls[1] > 32) {
      gs[1] = (ls[1] + 31) / 32;
      ls[1] = 32;
    }
    if (ls[0] * ls[1] > 512) {
      gs[0] = (ls[0] + 15) / 16;
      ls[0] = 16;
    }
  }
  if (gs[0] * gs[1] * gs[2] > 65535) {
    if (gs[0] * gs[1] > 65535)
      return GA_VALUE_ERROR;
    gs[2] = (65535 / (gs[0] * gs[1]));
  }

  if (order == cb_c) {
    t = M;
    M = N;
    N = t;
    tp = offX;
    offX = offY;
    offY = tp;
    t = incX;
    incX = incY;
    incY = t;
    T = x;
    x = y;
    y = T;
  }

  ASSERT_BUF(x[0]);

  ctx = x[0]->ctx;

  cuda_enter(ctx);

  {
    double **T_l = alloca(sizeof(double *) * batchCount * 3);
    const double **A_l = (const double **)T_l;
    const double **x_l = (const double **)T_l + batchCount;
    double **y_l = T_l + (batchCount * 2);

    for (i = 0; i < batchCount; i++) {
      ASSERT_BUF(A[i]);
      ASSERT_BUF(x[i]);
      ASSERT_BUF(y[i]);
      cuda_wait(A[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      cuda_wait(x[i], CUDA_WAIT_READ);
      cuda_wait(y[i], CUDA_WAIT_READ);
      A_l[i] = (double *)(A[i]->ptr + offA[i]);
      x_l[i] = (double *)(x[i]->ptr + offX[i]);
      y_l[i] = (double *)(y[i]->ptr + offY[i]);
    }

    Aa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, A_l,
                               GA_BUFFER_INIT, &err);
    if (Aa == NULL)
      return err;
    xa = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, x_l,
                               GA_BUFFER_INIT, &err);
    if (xa == NULL) {
      cuda_ops.buffer_release(Aa);
      return err;
    }
    ya = cuda_ops.buffer_alloc((gpucontext *)ctx, sizeof(double *) * batchCount, y_l,
                               GA_BUFFER_INIT, &err);
    if (ya == NULL) {
      cuda_ops.buffer_release(Aa);
      cuda_ops.buffer_release(xa);
      return err;
    }
  }

  args[0] = xa;
  args[1] = &incX;
  args[2] = ya;
  args[3] = &incY;
  args[4] = &alpha;
  args[5] = Aa;
  args[6] = &lda;
  args[7] = &batchCount;
  args[8] = &M;
  args[9] = &N;

  err = GpuKernel_call(&((blas_handle *)ctx->blas_handle)->sgerBH_gen_small, 3, ls, gs, 0, args);

  cuda_ops.buffer_release(Aa);
  cuda_ops.buffer_release(xa);
  cuda_ops.buffer_release(ya);

  if (err != GA_NO_ERROR) {
    cuda_exit(ctx);
    return err;
  }


  for (i = 0; i < batchCount; i++) {
    cuda_record(A[i], CUDA_WAIT_READ);
    cuda_record(x[i], CUDA_WAIT_READ);
    cuda_record(y[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
  }

  cuda_exit(ctx);
  return GA_NO_ERROR;
}

GPUARRAY_LOCAL gpuarray_blas_ops cublas_ops = {
  setup,
  teardown,
  error,
  hgemv, /* TODO */
  sgemv,
  dgemv,
  hgemm,
  sgemm,
  dgemm,
  hger, /* TODO */
  sger,
  dger,
  hgemmBatch, /* TODO */
  sgemmBatch,
  dgemmBatch,
  hgemvBatch, /* TODO */
  sgemvBatch,
  dgemvBatch,
  hgerBatch, /* TODO */
  sgerBatch,
  dgerBatch
};
