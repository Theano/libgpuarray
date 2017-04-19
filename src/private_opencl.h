#ifndef _GPUARRAY_PRIVATE_OPENCL
#define _GPUARRAY_PRIVATE_OPENCL

#include "private.h"

#include "loaders/libopencl.h"

#ifdef DEBUG
#include <assert.h>

#define CTX_TAG "ocl ctx "
#define BUF_TAG "ocl buf "
#define KER_TAG "ocl kern"

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

const char *cl_error_string(cl_int);

static inline int error_cl(error *e, const char *msg, cl_int err) {
  return error_fmt(e, GA_IMPL_ERROR, "%s: %s", msg, cl_error_string(err));
}

#define CL_CHECK(e, cmd) do {                   \
    cl_int err = (cmd);                         \
    if (err != CL_SUCCESS)                      \
      return error_cl(e, #cmd, err);            \
  } while(0)

#define CL_CHECKN(e, cmd) do {                  \
    cl_int err = (cmd);                         \
    if (err != CL_SUCCESS) {                    \
      error_cl(e, #cmd, err);                   \
      return NULL;                              \
    }                                           \
  } while(0)

#define CL_GET_PROP(e, fn, obj, prop, val) do {     \
    size_t sz;                                      \
    cl_int err;                                     \
    CL_CHECK(e, fn (obj, prop, 0, NULL, &sz));      \
    val = malloc(sz);                               \
    if (val == NULL) return error_sys(e, "malloc"); \
    err = fn (obj, prop, sz, val, NULL);            \
    if (err != CL_SUCCESS) {                        \
      free(val);                                    \
      val = NULL;                                   \
      return error_cl(e, #fn, err);                 \
    }                                               \
  } while(0)

typedef struct _cl_ctx {
  GPUCONTEXT_HEAD;
  cl_context ctx;
  cl_command_queue q;
  char *exts;
  char *preamble;
} cl_ctx;

STATIC_ASSERT(sizeof(cl_ctx) <= sizeof(gpucontext), sizeof_struct_gpucontext_cl);

struct _gpudata {
  cl_mem buf;
  cl_ctx *ctx;
  /* Don't change anyhting above this without checking
     struct _partial_gpudata */
  cl_event ev;
  unsigned int refcnt;
#ifdef DEBUG
  char tag[8];
#endif
};

struct _gpukernel {
  cl_ctx *ctx; /* Keep the context first */
  cl_kernel k;
  cl_event ev;
  cl_event **evr;
  int *types;
  unsigned int argcount;
  unsigned int refcnt;
  cl_uint num_ev;
#ifdef DEBUG
  char tag[8];
#endif
};

cl_ctx *cl_make_ctx(cl_context ctx, int flags);
cl_command_queue cl_get_stream(gpucontext *ctx);
gpudata *cl_make_buf(gpucontext *c, cl_mem buf);
cl_mem cl_get_buf(gpudata *g);

#endif
