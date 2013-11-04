#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "compyte/buffer.h"
#include "compyte/error.h"

const char *Gpu_error(const compyte_buffer_ops *o, void *ctx, int err) {
  if (err == GA_IMPL_ERROR)
    return o->ctx_error(ctx);
  else
    return compyte_error_str(err);
}

#ifdef WITH_CUDA
extern const compyte_buffer_ops cuda_ops;
#endif
#ifdef WITH_OPENCL
extern const compyte_buffer_ops opencl_ops;
#endif

const compyte_buffer_ops *compyte_get_ops(const char *name) {
#ifdef WITH_CUDA
  if (strcmp("cuda", name) == 0) return &cuda_ops;
#endif
#ifdef WITH_OPENCL
  if (strcmp("opencl", name) == 0) return &opencl_ops;
#endif
  return NULL;
}

#define FAIL(v, e) { if (ret) *ret = e; return v; }

gpudata *compyte_buffer_transfer(gpudata *buf, size_t offset, size_t sz,
                                 void *src_ctx,
                                 const compyte_buffer_ops *src_ops,
                                 void *dst_ctx,
                                 const compyte_buffer_ops *dst_ops,
                                 int may_share, int *ret) {
  gpudata *res;
  void *tmp;
  int err;

  if (src_ops == dst_ops) {
    res = src_ops->buffer_transfer(buf, offset, sz, dst_ctx, may_share);
    if (res != NULL)
      return res;
  }

  /* Fallback to a host memory copy */
  tmp = malloc(sz);
  if (tmp == NULL)
    FAIL(NULL, GA_MEMORY_ERROR);
  err = src_ops->buffer_read(tmp, buf, offset, sz);
  if (err != GA_NO_ERROR) {
    free(tmp);
    FAIL(NULL, err);
  }
  res = dst_ops->buffer_alloc(dst_ctx, sz, tmp, GA_BUFFER_INIT, ret);
  free(tmp);
  return res;
}
