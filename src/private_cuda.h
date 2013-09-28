#ifndef _PRIVATE_CUDA_H
#define _PRIVATE_CUDA_H

#ifdef __APPLE__
#include <CUDA/cuda.h>
#else
#include <cuda.h>
#endif

#include "private.h"

#include "compyte/buffer.h"

#define DONTFREE 0x1000

typedef struct _cuda_context {
  CUcontext ctx;
  CUcontext old;
  CUresult err;
  CUstream s;
  void *blas_handle;
  unsigned int refcnt;
  int flags;
} cuda_context;

COMPYTE_LOCAL void *cuda_make_ctx(CUcontext ctx, int flags);
COMPYTE_LOCAL CUcontext cuda_get_ctx(void *ctx);
COMPYTE_LOCAL CUstream cuda_get_stream(void *ctx);
COMPYTE_LOCAL void cuda_enter(cuda_context *ctx);
COMPYTE_LOCAL void cuda_exit(cuda_context *ctx);

struct _gpudata {
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

/* The total size of the arguments is limited to 256 bytes */
#define NUM_ARGS (256/sizeof(void*))

struct _gpukernel {
  CUmodule m;
  CUfunction k;
  void *args[NUM_ARGS];
#if CUDA_VERSION < 4000
  size_t types[NUM_ARGS];
#endif
  unsigned int argcount;
  gpudata *bs[NUM_ARGS];
  cuda_context *ctx;
  unsigned int refcnt;
};

#endif
