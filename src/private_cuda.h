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
#define COMM_TAG "cudacomm"

#define TAG_CTX(c) memcpy((c)->tag, CTX_TAG, 8)
#define TAG_BUF(b) memcpy((b)->tag, BUF_TAG, 8)
#define TAG_KER(k) memcpy((k)->tag, KER_TAG, 8)
#define TAG_COMM(co) memcpy((co)->tag, COMM_TAG, 8)
#define ASSERT_CTX(c) assert(memcmp((c)->tag, CTX_TAG, 8) == 0)
#define ASSERT_BUF(b) assert(memcmp((b)->tag, BUF_TAG, 8) == 0)
#define ASSERT_KER(k) assert(memcmp((k)->tag, KER_TAG, 8) == 0)
#define ASSERT_COMM(co) assert(memcmp((co)->tag, COMM_TAG, 8) == 0)
#define CLEAR(o) memset((o)->tag, 0, 8);

#else
#define TAG_CTX(c)
#define TAG_BUF(b)
#define TAG_KER(k)
#define TAG_COMM(k)
#define ASSERT_CTX(c)
#define ASSERT_BUF(b)
#define ASSERT_KER(k)
#define ASSERT_COMM(k)
#define CLEAR(o)
#endif

/* Keep in sync with the copy in gpuarray/extension.h */
#define DONTFREE 0x10000000

typedef struct _cuda_context {
  GPUCONTEXT_HEAD;
  CUcontext ctx;
  CUresult err;
  CUstream s;
  CUstream mem_s;
  gpudata *freeblocks;
  cache *kernel_cache;
  unsigned int enter;
} cuda_context;

STATIC_ASSERT(sizeof(cuda_context) <= sizeof(gpucontext), sizeof_struct_gpucontext_cuda);


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

GPUARRAY_LOCAL cuda_context *cuda_make_ctx(CUcontext ctx, int flags);
GPUARRAY_LOCAL CUstream cuda_get_stream(cuda_context *ctx);
GPUARRAY_LOCAL void cuda_enter(cuda_context *ctx);
GPUARRAY_LOCAL void cuda_exit(cuda_context *ctx);

struct _gpudata {
  CUdeviceptr ptr;
  cuda_context *ctx;
  /* Don't change anything abovbe this without checking
     struct _partial_gpudata */
  CUevent rev;
  CUevent wev;
  CUstream ls; /* last stream used */
  unsigned int refcnt;
  int flags;
  size_t sz;
  gpudata *next;
#ifdef DEBUG
  char tag[8];
#endif
};

GPUARRAY_LOCAL gpudata *cuda_make_buf(cuda_context *c, CUdeviceptr p, size_t sz);
GPUARRAY_LOCAL CUdeviceptr cuda_get_ptr(gpudata *g);
GPUARRAY_LOCAL size_t cuda_get_sz(gpudata *g);
GPUARRAY_LOCAL int cuda_wait(gpudata *, int);
GPUARRAY_LOCAL int cuda_record(gpudata *, int);

/* private flags are in the upper 16 bits */
#define CUDA_WAIT_READ  0x10000
#define CUDA_WAIT_WRITE 0x20000
#define CUDA_WAIT_FORCE 0x40000

#define CUDA_WAIT_ALL   (CUDA_WAIT_READ|CUDA_WAIT_WRITE)

#define CUDA_HEAD_ALLOC 0x40000
#define CUDA_MAPPED_PTR 0x80000

struct _gpukernel {
  cuda_context *ctx; /* Keep the context first */
  CUmodule m;
  CUfunction k;
  void **args;
  size_t bin_sz;
  void *bin;
  int *types;
  unsigned int argcount;
  unsigned int refcnt;
#ifdef DEBUG
  char tag[8];
#endif
};

#endif
