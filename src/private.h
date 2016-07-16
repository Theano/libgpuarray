#ifndef _PRIVATE
#define _PRIVATE

/** \cond INTERNAL_DOCS */

/*
 * This file contains function definition that are shared in multiple
 * files but not exposed in the interface.
 */

#include "private_config.h"

#include <gpuarray/array.h>
#include <gpuarray/types.h>
#include <gpuarray/buffer.h>
#include <gpuarray/buffer_blas.h>
#include <gpuarray/buffer_collectives.h>

#include "util/strb.h"
#include "cache.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#define ADDR32_MAX   4294967295
#define SADDR32_MIN -2147483648
#define SADDR32_MAX  2147483647

struct _gpuarray_buffer_ops;
typedef struct _gpuarray_buffer_ops gpuarray_buffer_ops;

struct _gpuarray_blas_ops;
typedef struct _gpuarray_blas_ops gpuarray_blas_ops;

struct _gpuarray_comm_ops;
typedef struct _gpuarray_comm_ops gpuarray_comm_ops;

#define GPUCONTEXT_HEAD                         \
  const gpuarray_buffer_ops *ops;               \
  const gpuarray_blas_ops *blas_ops;            \
  const gpuarray_comm_ops *comm_ops;            \
  void *blas_handle;                            \
  const char* error_msg;                        \
  unsigned int refcnt;                          \
  int flags;                                    \
  struct _gpudata *errbuf;                      \
  cache *extcopy_cache;                         \
  char bin_id[64];                              \
  char tag[8]

struct _gpucontext {
  GPUCONTEXT_HEAD;
  void *ctx_ptr;
  void *private[7];
};

/* The real gpudata struct is likely bigger but we only care about the
   first two members for now. */
typedef struct _partial_gpudata {
  void *devptr;
  gpucontext *ctx;
} partial_gpudata;

typedef struct _partial_gpukernel {
  gpucontext *ctx;
} partial_gpukernel;

typedef struct _partial_gpucomm {
  gpucontext* ctx;
} partial_gpucomm;

struct _gpuarray_buffer_ops {
  int (*get_platform_count)(unsigned int* platcount);
  int (*get_device_count)(unsigned int platform, unsigned int* devcount);
  gpucontext *(*buffer_init)(int dev, int flags, int *ret);
  void (*buffer_deinit)(gpucontext *ctx);
  gpudata *(*buffer_alloc)(gpucontext *ctx, size_t sz, void *data, int flags,
                           int *ret);
  void (*buffer_retain)(gpudata *b);
  void (*buffer_release)(gpudata *b);
  int (*buffer_share)(gpudata *a, gpudata *b, int *ret);
  int (*buffer_move)(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                     size_t sz);
  int (*buffer_read)(void *dst, gpudata *src, size_t srcoff, size_t sz);
  int (*buffer_write)(gpudata *dst, size_t dstoff, const void *src, size_t sz);
  int (*buffer_memset)(gpudata *dst, size_t dstoff, int data);
  gpukernel *(*kernel_alloc)(gpucontext *ctx, unsigned int count,
                             const char **strings, const size_t *lengths,
                             const char *fname, unsigned int numargs,
                             const int *typecodes, int flags, int *ret,
                             char **err_str);
  void (*kernel_retain)(gpukernel *k);
  void (*kernel_release)(gpukernel *k);
  int (*kernel_setarg)(gpukernel *k, unsigned int i, void *a);
  int (*kernel_call)(gpukernel *k, unsigned int n,
                     const size_t *bs, const size_t *gs,
                     size_t shared, void **args);

  int (*kernel_binary)(gpukernel *k, size_t *sz, void **obj);
  int (*buffer_sync)(gpudata *b);
  int (*buffer_transfer)(gpudata *dst, size_t dstoff,
                         gpudata *src, size_t srcoff, size_t sz);
  int (*property)(gpucontext *ctx, gpudata *buf, gpukernel *k, int prop_id,
                  void *res);
  const char *(*ctx_error)(gpucontext *ctx);
};

struct _gpuarray_blas_ops {
  int (*setup)(gpucontext *ctx);
  void (*teardown)(gpucontext *ctx);
  const char *(*error)(gpucontext *ctx);
  int (*hgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               float alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, float beta,
               gpudata *Y, size_t offY, int incY);
  int (*sgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               float alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, float beta,
               gpudata *Y, size_t offY, int incY);
  int (*dgemv)(cb_order order, cb_transpose transA, size_t M, size_t N,
               double alpha, gpudata *A, size_t offA, size_t lda,
               gpudata *X, size_t offX, int incX, double beta,
               gpudata *Y, size_t offY, int incY);
  int (*hgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, float alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               float beta, gpudata *C, size_t offC, size_t ldc);
  int (*sgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, float alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               float beta, gpudata *C, size_t offC, size_t ldc);
  int (*dgemm)(cb_order order, cb_transpose transA, cb_transpose transB,
               size_t M, size_t N, size_t K, double alpha,
               gpudata *A, size_t offA, size_t lda,
               gpudata *B, size_t offB, size_t ldb,
               double beta, gpudata *C, size_t offC, size_t ldc);
  int (*hger)(cb_order order, size_t M, size_t N, float alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*sger)(cb_order order, size_t M, size_t N, float alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*dger)(cb_order order, size_t M, size_t N, double alpha,
              gpudata *X, size_t offX, int incX,
              gpudata *Y, size_t offY, int incY,
              gpudata *A, size_t offA, size_t lda);
  int (*hgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    float beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*sgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    float beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*dgemmBatch)(cb_order order, cb_transpose transA, cb_transpose transB,
                    size_t M, size_t N, size_t K, double alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **B, size_t *offB, size_t ldb,
                    double beta, gpudata **C, size_t *offC, size_t ldc,
                    size_t batchCount);
  int (*hgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    float beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*sgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, float alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    float beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*dgemvBatch)(cb_order order, cb_transpose transA,
                    size_t M, size_t N, double alpha,
                    gpudata **A, size_t *offA, size_t lda,
                    gpudata **x, size_t *offX, size_t incX,
                    double beta, gpudata **y, size_t *offY, size_t incY,
                    size_t batchCount, int flags);
  int (*hgerBatch)(cb_order order, size_t M, size_t N, float alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
  int (*sgerBatch)(cb_order order, size_t M, size_t N, float alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
  int (*dgerBatch)(cb_order order, size_t M, size_t N, double alpha,
                   gpudata **x, size_t *offX, size_t incX,
                   gpudata **y, size_t *offY, size_t incY,
                   gpudata **A, size_t *offA, size_t lda,
                   size_t batchCount, int flags);
};

struct _gpuarray_comm_ops {
  int (*comm_new)(gpucomm** comm, gpucontext* ctx, gpucommCliqueId comm_id,
                  int ndev, int rank);
  void (*comm_free)(gpucomm* comm);
  int (*generate_clique_id)(gpucontext* ctx, gpucommCliqueId* comm_id);
  int (*get_count)(const gpucomm* comm, int* count);
  int (*get_rank)(const gpucomm* comm, int* rank);
  // collective ops
  int (*reduce)(gpudata* src, size_t offsrc,
                gpudata* dest, size_t offdest,
                size_t count, int typecode, int opcode,
                int root, gpucomm* comm);
  int (*all_reduce)(gpudata* src, size_t offsrc,
                    gpudata* dest, size_t offdest,
                    size_t count, int typecode, int opcode,
                    gpucomm* comm);
  int (*reduce_scatter)(gpudata* src, size_t offsrc,
                        gpudata* dest, size_t offdest,
                        size_t count, int typecode, int opcode,
                        gpucomm* comm);
  int (*broadcast)(gpudata* array, size_t offset,
                   size_t count, int typecode,
                   int root, gpucomm* comm);
  int (*all_gather)(gpudata* src, size_t offsrc,
                    gpudata* dest, size_t offdest,
                    size_t count, int typecode,
                    gpucomm* comm);
};

#define STATIC_ASSERT(COND, MSG) typedef char static_assertion_##MSG[2*(!!(COND))-1]

static inline void *memdup(const void *p, size_t s) {
  void *res = malloc(s);
  if (res != NULL)
    memcpy(res, p, s);
  return res;
}

GPUARRAY_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_aligned(const GpuArray *a);

GPUARRAY_LOCAL extern const gpuarray_type scalar_types[];
GPUARRAY_LOCAL extern const gpuarray_type vector_types[];

/*
 * This function generates the kernel code to perform indexing on var id
 * from planar index 'i' using the dimensions and strides provided.
 */
GPUARRAY_LOCAL void gpuarray_elem_perdim(strb *sb, unsigned int nd,
                                         const size_t *dims,
                                         const ssize_t *str,
                                         const char *id);

GPUARRAY_LOCAL void gpukernel_source_with_line_numbers(unsigned int count,
                                                       const char **news,
                                                       size_t *newl,
                                                       strb *src);

#define ISSET(v, fl) ((v) & (fl))
#define ISCLR(v, fl) (!((v) & (fl)))

#define FLSET(v, fl) (v |= (fl))
#define FLCLR(v, fl) (v &= ~(fl))

#define GA_CHECK(cmd)       \
  do {                      \
    int err = (cmd);        \
    if (err != GA_NO_ERROR) \
      return err;           \
  } while (0)

#ifdef __cplusplus
}
#endif

/** \endcond */

#endif
