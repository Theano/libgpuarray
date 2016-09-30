#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

#include "private.h"

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

int gpu_get_platform_count(const char* name, unsigned int* platcount) {
  const gpuarray_buffer_ops* ops = gpuarray_get_ops(name);
  if (ops == NULL) {
    return GA_INVALID_ERROR;
  }
  return ops->get_platform_count(platcount);
}

int gpu_get_device_count(const char* name, unsigned int platform,
                         unsigned int* devcount) {
  const gpuarray_buffer_ops* ops = gpuarray_get_ops(name);
  if (ops == NULL) {
    return GA_INVALID_ERROR;
  }
  return ops->get_device_count(platform, devcount);
}

gpucontext *gpucontext_init(const char *name, int dev, int flags, int *ret) {
  gpucontext *res;
  const gpuarray_buffer_ops *ops = gpuarray_get_ops(name);
  if (ops == NULL)
    FAIL(NULL, GA_INVALID_ERROR);
  res = ops->buffer_init(dev, flags, ret);
  if (res == NULL)
    return NULL;
  res->ops = ops;
  if (gpucontext_property(res, GA_CTX_PROP_BLAS_OPS, &res->blas_ops) != GA_NO_ERROR)
    res->blas_ops = NULL;
  res->blas_handle = NULL;
  if (gpucontext_property(res, GA_CTX_PROP_COMM_OPS, &res->comm_ops) != GA_NO_ERROR)
    res->comm_ops = NULL;
  res->extcopy_cache = NULL;
  return res;
}

void gpucontext_deref(gpucontext *ctx) {
  if (ctx->blas_handle != NULL)
    ctx->blas_ops->teardown(ctx);
  if (ctx->extcopy_cache != NULL) {
    cache_destroy(ctx->extcopy_cache);
    ctx->extcopy_cache = NULL;
  }
  ctx->ops->buffer_deinit(ctx);
}

int gpucontext_property(gpucontext *ctx, int prop_id, void *res) {
  return ctx->ops->property(ctx, NULL, NULL, prop_id, res);
}

const char *gpucontext_error(gpucontext *ctx, int err) {
  if (ctx != NULL) {
    switch (err) {
    case GA_IMPL_ERROR:
      return ctx->ops->ctx_error(ctx);
    case GA_BLAS_ERROR:
      return gpublas_error(ctx);
    case GA_COMM_ERROR:
      return gpucomm_error(ctx);
    }
  }
  return gpuarray_error_str(err);
}

gpudata *gpudata_alloc(gpucontext *ctx, size_t sz, void *data, int flags,
                       int *ret) {
  return ctx->ops->buffer_alloc(ctx, sz, data, flags, ret);
}

void gpudata_retain(gpudata *b) {
  ((partial_gpudata *)b)->ctx->ops->buffer_retain(b);
}

void gpudata_release(gpudata *b) {
  if(b){
    ((partial_gpudata *)b)->ctx->ops->buffer_release(b);
  }
}

int gpudata_share(gpudata *a, gpudata *b, int *ret) {
  return ((partial_gpudata *)a)->ctx->ops->buffer_share(a, b, ret);
}

int gpudata_move(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                 size_t sz) {
  return ((partial_gpudata *)src)->ctx->ops->buffer_move(dst, dstoff,
                                                         src, srcoff, sz);
}

int gpudata_transfer(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                     size_t sz) {
  gpucontext *src_ctx;
  gpucontext *dst_ctx;
  void *tmp;
  int res;
  src_ctx = ((partial_gpudata *)src)->ctx;
  dst_ctx = ((partial_gpudata *)dst)->ctx;
  if (src_ctx == dst_ctx)
    return src_ctx->ops->buffer_move(dst, dstoff, src, srcoff, sz);
  if (src_ctx->ops == dst_ctx->ops) {
    res = src_ctx->ops->buffer_transfer(dst, dstoff, src, srcoff, sz);
    if (res == GA_NO_ERROR)
      return res;
  }

  /* Fallback to host copy */
  tmp = malloc(sz);
  if (tmp == NULL)
    return GA_MEMORY_ERROR;
  res = src_ctx->ops->buffer_read(tmp, src, srcoff, sz);
  if (res != GA_NO_ERROR) {
    free(tmp);
    return res;
  }
  res = dst_ctx->ops->buffer_write(dst, dstoff, tmp, sz);
  free(tmp);
  return res;
}

int gpudata_read(void *dst, gpudata *src, size_t srcoff, size_t sz) {
  return ((partial_gpudata *)src)->ctx->ops->buffer_read(dst, src, srcoff, sz);
}

int gpudata_write(gpudata *dst, size_t dstoff, const void *src, size_t sz) {
  return ((partial_gpudata *)dst)->ctx->ops->buffer_write(dst, dstoff,
                                                          src, sz);
}

int gpudata_memset(gpudata *dst, size_t dstoff, int data) {
  return ((partial_gpudata *)dst)->ctx->ops->buffer_memset(dst, dstoff, data);
}

int gpudata_sync(gpudata *b) {
  return ((partial_gpudata *)b)->ctx->ops->buffer_sync(b);
}

int gpudata_property(gpudata *b, int prop_id, void *res) {
  return ((partial_gpudata *)b)->ctx->ops->property(NULL, b, NULL, prop_id,
                                                    res);
}

gpukernel *gpukernel_init(gpucontext *ctx, unsigned int count,
                          const char **strings, const size_t *lengths,
                          const char *fname, unsigned int numargs,
                          const int *typecodes, int flags, int *ret,
                          char **err_str) {
  return ctx->ops->kernel_alloc(ctx, count, strings, lengths, fname, numargs,
                                typecodes, flags, ret, err_str);
}

void gpukernel_retain(gpukernel *k) {
  ((partial_gpukernel *)k)->ctx->ops->kernel_retain(k);
}

void gpukernel_release(gpukernel *k) {
  ((partial_gpukernel *)k)->ctx->ops->kernel_release(k);
}

int gpukernel_setarg(gpukernel *k, unsigned int i, void *a) {
  return ((partial_gpukernel *)k)->ctx->ops->kernel_setarg(k, i, a);
}

int gpukernel_call(gpukernel *k, unsigned int n, const size_t *ls,
                   const size_t *gs, size_t shared, void **args) {
  return ((partial_gpukernel *)k)->ctx->ops->kernel_call(k, n, ls, gs,
                                                         shared, args);
}

int gpukernel_binary(gpukernel *k, size_t *sz, void **obj) {
  return ((partial_gpukernel *)k)->ctx->ops->kernel_binary(k, sz, obj);
}

int gpukernel_property(gpukernel *k, int prop_id, void *res) {
  return ((partial_gpukernel *)k)->ctx->ops->property(NULL, NULL, k, prop_id,
                                                      res);
}

gpucontext *gpudata_context(gpudata *b) {
  return ((partial_gpudata *)b)->ctx;
}

gpucontext *gpukernel_context(gpukernel *k) {
  return ((partial_gpukernel *)k)->ctx;
}
