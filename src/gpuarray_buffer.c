#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "gpuarray/buffer.h"
#include "gpuarray/error.h"

const char *Gpu_error(const gpuarray_buffer_ops *o, gpucontext *ctx, int err) {
  if (err == GA_IMPL_ERROR)
    return o->ctx_error(ctx);
  else
    return gpuarray_error_str(err);
}

#ifdef WITH_CUDA
extern const gpuarray_buffer_ops cuda_ops;
#endif
#ifdef WITH_OPENCL
extern const gpuarray_buffer_ops opencl_ops;
#endif

const gpuarray_buffer_ops *gpuarray_get_ops(const char *name) {
#ifdef WITH_CUDA
  if (strcmp("cuda", name) == 0) return &cuda_ops;
#endif
#ifdef WITH_OPENCL
  if (strcmp("opencl", name) == 0) return &opencl_ops;
#endif
  return NULL;
}

#define FAIL(v, e) { if (ret) *ret = e; return v; }

int gpucontext_init(gpucontext **res, const gpuarray_buffer_ops *ops,
                    int dev, int flags) {
  int ret;
  *res = ops->buffer_init(dev, flags, &ret);
  return ret;
}




gpudata *gpuarray_buffer_transfer(gpudata *buf, size_t offset, size_t sz,
                                  gpucontext *src_ctx,
                                  const gpuarray_buffer_ops *src_ops,
                                  gpucontext *dst_ctx,
                                  const gpuarray_buffer_ops *dst_ops,
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

gpucontext *gpuarray_buffer_context(const gpuarray_buffer_ops *ops, gpudata *b) {
  gpucontext *ctx;
  int err = ops->property(NULL, b, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    return NULL;
  return ctx;
}
