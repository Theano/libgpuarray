#ifndef _COMPYTE_PRIVATE_OPENCL
#define _COMPYTE_PRIVATE_OPENCL

#include "private.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

typedef struct _cl_ctx {
  cl_context ctx;
  cl_command_queue q;
  char *exts;
  void *blas_handle;
  cl_int err;
  unsigned int refcnt;
} cl_ctx;

struct _gpudata {
  cl_mem buf;
  cl_event ev;
  cl_ctx *ctx;
  cl_uint refcnt;
};

struct _gpukernel {
  cl_kernel k;
  cl_event ev;
  gpudata **bs;
  cl_ctx *ctx;
  unsigned int refcnt;
};

COMPYTE_LOCAL cl_ctx *cl_make_ctx(cl_context ctx);
COMPYTE_LOCAL cl_context cl_get_ctx(void *ctx);
COMPYTE_LOCAL cl_command_queue cl_get_stream(void *ctx);
COMPYTE_LOCAL gpudata *cl_make_buf(void *c, cl_mem buf);
COMPYTE_LOCAL cl_mem cl_get_buf(gpudata *g);

#endif
