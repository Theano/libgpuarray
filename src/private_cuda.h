#ifndef _PRIVATE_CUDA_H
#define _PRIVATE_CUDA_H

#ifdef __APPLE__
#include <CUDA/cuda.h>
#else
#include <cuda.h>
#endif

#include <cache.h>

#include "private.h"

#include "gpuarray/buffer.h"

#ifdef DEBUG
#include <assert.h>

#define CTX_TAG "cudactx "
#define BUF_TAG "cudabuf "
#define KER_TAG "cudakern"

#define TAG_CTX(c) memcpy((c)->tag, CTX_TAG, 8)
#define TAG_BUF(b) memcpy((b)->tag, BUF_TAG, 8)
#define TAG_KER(k) memcpy((k)->tag, KER_TAG, 8)
#define ASSERT_CTX(c) assert(memcmp((c)->tag, CTX_TAG, 8) == 0)
#define ASSERT_BUF(b) assert(memcmp((b)->tag, BUF_TAG, 8) == 0)
#define ASSERT_KER(k) assert(memcmp((k)->tag, KER_TAG, 8) == 0)
#define CLEAR(o) memset((o)->tag, 0, 8);

#else
#define TAG_CTX(c)
#define TAG_BUF(b)
#define TAG_KER(k)
#define ASSERT_CTX(c)
#define ASSERT_BUF(b)
#define ASSERT_KER(k)
#define CLEAR(o)
#endif

/* Keep in sync with the copy in gpuarray/extension.h */
#define DONTFREE 0x10000000

#define BIN_ID_LEN 12

typedef struct _cuda_context {
#ifdef DEBUG
  char tag[8];
#endif
  CUcontext ctx;
  CUresult err;
  CUstream s;
  CUstream mem_s;
  void *blas_handle;
  gpudata *errbuf;
  cache *extcopy_cache;
  char bin_id[BIN_ID_LEN];
  unsigned int refcnt;
  int flags;
  unsigned int enter;
  gpudata *freeblocks;
} cuda_context;

/*
 * About freeblocks.
 *
 * Freeblocks is a linked list of gpudata instances that are
 * considrered to be "free".  That is they are not in use anywhere
 * else in the program.  It is used to cache and reuse allocations so
 * that we can avoid the heavy cost and synchronization of
 * cuMemAlloc() and cuMemFree().
 *
 * It is ordered by pointer address.  When adding back to it, blocks
 * will be merged with their neighbours, but not across original
 * allocation lines (which are kept track of with the CUDA_HEAD_ALLOC
 * flag.
 */

#ifdef WITH_NVRTC
#define ARCH_PREFIX "compute_"
#else
#define ARCH_PREFIX "sm_"
#endif

GPUARRAY_LOCAL void *cuda_make_ctx(CUcontext ctx, int flags);
GPUARRAY_LOCAL CUcontext cuda_get_ctx(void *ctx);
GPUARRAY_LOCAL CUstream cuda_get_stream(void *ctx);
GPUARRAY_LOCAL void cuda_enter(cuda_context *ctx);
GPUARRAY_LOCAL void cuda_exit(cuda_context *ctx);

struct _gpudata {
  CUdeviceptr ptr;
  CUevent ev;
  size_t sz;
  cuda_context *ctx;
  int flags;
  unsigned int refcnt;
  gpudata *next;
#ifdef DEBUG
  char tag[8];
#endif
};

GPUARRAY_LOCAL gpudata *cuda_make_buf(void *c, CUdeviceptr p, size_t sz);
GPUARRAY_LOCAL CUdeviceptr cuda_get_ptr(gpudata *g);
GPUARRAY_LOCAL size_t cuda_get_sz(gpudata *g);
GPUARRAY_LOCAL int cuda_wait(gpudata *, int);
GPUARRAY_LOCAL int cuda_record(gpudata *, int);

/* private flags are in the upper 16 bits */
#define CUDA_WAIT_READ  0x10000
#define CUDA_WAIT_WRITE 0x20000
#define CUDA_WAIT_MASK  0x30000

#define CUDA_WAIT_ALL   (CUDA_WAIT_READ|CUDA_WAIT_WRITE)

#define CUDA_HEAD_ALLOC 0x40000
#define CUDA_MAPPED_PTR 0x80000

struct _gpukernel {
#ifdef DEBUG
  char tag[8];
#endif
  cuda_context *ctx;
  CUmodule m;
  CUfunction k;
  void **args;
  size_t bin_sz;
  void *bin;
  int *types;
  unsigned int argcount;
  unsigned int refcnt;
};

#endif
