#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

#include "util/error.h"
#include "private.h"

extern const gpuarray_buffer_ops cuda_ops;
extern const gpuarray_buffer_ops opencl_ops;

const gpuarray_buffer_ops *gpuarray_get_ops(const char *name) {
  if (strcmp("cuda", name) == 0) return &cuda_ops;
  if (strcmp("opencl", name) == 0) return &opencl_ops;
  return NULL;
}

#define FAIL(v, e) { if (ret) *ret = (e)->code; return v; }

int gpu_get_platform_count(const char* name, unsigned int* platcount) {
  const gpuarray_buffer_ops* ops = gpuarray_get_ops(name);
  if (ops == NULL) {
    return error_set(global_err, GA_INVALID_ERROR, "Invalid platform");
  }
  return ops->get_platform_count(platcount);
}

int gpu_get_device_count(const char* name, unsigned int platform,
                         unsigned int* devcount) {
  const gpuarray_buffer_ops* ops = gpuarray_get_ops(name);
  if (ops == NULL) {
    return error_set(global_err, GA_INVALID_ERROR, "Invalid platform");
  }
  return ops->get_device_count(platform, devcount);
}

int gpucontext_props_new(gpucontext_props **res) {
  gpucontext_props *r = calloc(1, sizeof(gpucontext_props));
  if (r == NULL) return error_sys(global_err, "calloc");
  r->dev = -1;
  r->sched = GA_CTX_SCHED_AUTO;
  r->flags = 0;
  r->kernel_cache_path = NULL;
  r->initial_cache_size = 0;
  r->max_cache_size = (size_t)-1;
  *res = r;
  return GA_NO_ERROR;
}

int gpucontext_props_cuda_dev(gpucontext_props *p, int devno) {
  p->dev = devno;
  return GA_NO_ERROR;
}

int gpucontext_props_opencl_dev(gpucontext_props *p, int platno, int devno) {
  p->dev = (platno << 16) | devno;
  return GA_NO_ERROR;
}

int gpucontext_props_sched(gpucontext_props *p, int sched) {
  switch (sched) {
  case GA_CTX_SCHED_MULTI:
  case GA_CTX_SCHED_AUTO:
  case GA_CTX_SCHED_SINGLE:
    p->sched = sched;
    break;
  default:
    return error_fmt(global_err, GA_INVALID_ERROR, "Invalid value for sched: %d", sched);
  }

  if (sched == GA_CTX_SCHED_MULTI)
    FLSET(p->flags, GA_CTX_MULTI_THREAD);
  else
    FLCLR(p->flags, GA_CTX_MULTI_THREAD);

  return GA_NO_ERROR;
}

int gpucontext_props_set_single_stream(gpucontext_props *p) {
  p->flags |= GA_CTX_SINGLE_STREAM;
  return GA_NO_ERROR;
}

int gpucontext_props_kernel_cache(gpucontext_props *p, const char *path) {
  p->kernel_cache_path = path;
  return GA_NO_ERROR;
}

int gpucontext_props_alloc_cache(gpucontext_props *p, size_t initial, size_t max) {
  if (initial > max)
    return error_set(global_err, GA_VALUE_ERROR, "Initial size can't be bigger than max size");
  p->initial_cache_size = initial;
  p->max_cache_size = max;
  return GA_NO_ERROR;
}

void gpucontext_props_del(gpucontext_props *p) {
  free(p);
}

int gpucontext_init(gpucontext **res, const char *name, gpucontext_props *p) {
  const gpuarray_buffer_ops *ops = gpuarray_get_ops(name);
  gpucontext *r;
  if (ops == NULL) {
    gpucontext_props_del(p);
    return global_err->code;
  }
  if (p == NULL && gpucontext_props_new(&p) != GA_NO_ERROR)
    return global_err->code;
  r = ops->buffer_init(p);
  gpucontext_props_del(p);
  if (r == NULL) return global_err->code;
  r->ops = ops;
  r->extcopy_cache = NULL;
  *res = r;
  return GA_NO_ERROR;
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
  if (ctx == NULL)
    return global_err->msg;
  else
    return ctx->ops->ctx_error(ctx);
}

gpudata *gpudata_alloc(gpucontext *ctx, size_t sz, void *data, int flags,
                       int *ret) {
  gpudata *res = ctx->ops->buffer_alloc(ctx, sz, data, flags);
  if (res == NULL && ret) *ret = ctx->err->code;
  return res;
}

void gpudata_retain(gpudata *b) {
  ((partial_gpudata *)b)->ctx->ops->buffer_retain(b);
}

void gpudata_release(gpudata *b) {
  if (b)
    ((partial_gpudata *)b)->ctx->ops->buffer_release(b);
}

int gpudata_share(gpudata *a, gpudata *b, int *ret) {
  int res = ((partial_gpudata *)a)->ctx->ops->buffer_share(a, b);
  if (res == -1 && ret)
    *ret = ((partial_gpudata *)a)->ctx->err->code;
  return res;
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
  if (tmp == NULL) {
    error_sys(src_ctx->err, "malloc");
    return error_sys(dst_ctx->err, "malloc");
  }
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
  gpukernel *res = NULL;
  int err;
  err = ctx->ops->kernel_alloc(&res, ctx, count, strings, lengths, fname,
                               numargs, typecodes, flags, err_str);
  if (err != GA_NO_ERROR && ret != NULL)
    *ret = ctx->err->code;
  return res;
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

int gpukernel_call(gpukernel *k, unsigned int n, const size_t *gs,
                   const size_t *ls, size_t shared, void **args) {
  return ((partial_gpukernel *)k)->ctx->ops->kernel_call(k, n, gs, ls,
                                                         shared, args);
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
