#ifndef _PRIVATE_CUDA_H
#define _PRIVATE_CUDA_H

#ifdef __APPLE__
#include <CUDA/cuda.h>
#else
#include <cuda.h>
#endif

#include "private.h"

#include "compyte/buffer.h"

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


#define DONTFREE 0x10000000

typedef struct _cache cache;

typedef struct _cuda_context {
#ifdef DEBUG
  char tag[8];
#endif
  CUcontext ctx;
  CUcontext old;
  CUresult err;
  CUstream s;
  void *blas_handle;
  unsigned int refcnt;
  int flags;
  cache *extcopy_cache;
} cuda_context;

COMPYTE_LOCAL void *cuda_make_ctx(CUcontext ctx, int flags);
COMPYTE_LOCAL CUcontext cuda_get_ctx(void *ctx);
COMPYTE_LOCAL CUstream cuda_get_stream(void *ctx);
COMPYTE_LOCAL void cuda_enter(cuda_context *ctx);
COMPYTE_LOCAL void cuda_exit(cuda_context *ctx);

struct _gpudata {
#ifdef DEBUG
  char tag[8];
#endif
  CUdeviceptr ptr;
  CUevent ev;
  size_t sz;
  cuda_context *ctx;
  int flags;
  unsigned int refcnt;
};

COMPYTE_LOCAL gpudata *cuda_make_buf(void *c, CUdeviceptr p, size_t sz);
COMPYTE_LOCAL CUdeviceptr cuda_get_ptr(gpudata *g);
COMPYTE_LOCAL size_t cuda_get_sz(gpudata *g);

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
