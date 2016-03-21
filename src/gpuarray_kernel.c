#include "gpuarray/kernel.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

#include <stdlib.h>

int GpuKernel_init(GpuKernel *k, const gpuarray_buffer_ops *ops, void *ctx,
                   unsigned int count, const char **strs, const size_t *lens,
                   const char *name, unsigned int argcount, const int *types,
                   int flags, char **err_str) {
  int res = GA_NO_ERROR;

  k->args = calloc(argcount, sizeof(void *));
  if (k->args == NULL)
    return GA_MEMORY_ERROR;
  k->ops = ops;
  k->k = k->ops->kernel_alloc(ctx, count, strs, lens, name, argcount, types,
                              flags, &res, err_str);
  if (res != GA_NO_ERROR)
    GpuKernel_clear(k);
  return res;
}

void GpuKernel_clear(GpuKernel *k) {
  if (k->k)
    k->ops->kernel_release(k->k);
  free(k->args);
  k->k = NULL;
  k->ops = NULL;
  k->args = NULL;
}

void *GpuKernel_context(GpuKernel *k) {
  void *res = NULL;
  (void)k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_CTX, &res);
  return res;
}

int GpuKernel_sched(GpuKernel *k, size_t n, size_t *ls, size_t *gs) {
  size_t min_l;
  size_t max_l;
  size_t max_g;
  int err;

  err = k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_MAXLSIZE, &max_l);
  if (err != GA_NO_ERROR)
    return err;
  err = k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_PREFLSIZE, &min_l);
  if (err != GA_NO_ERROR)
    return err;
  err = k->ops->property(GpuKernel_context(k), NULL, NULL,
                         GA_CTX_PROP_MAXGSIZE, &max_g);
  if (err != GA_NO_ERROR)
    return err;
  if (*gs == 0) {
    if (*ls == 0) {
      if (n < max_l)
        *ls = min_l;
      else
        *ls = max_l;
    }
    *gs = ((n-1) / (*ls)) + 1;
    if (*gs > max_g)
      *gs = max_g;
  } else if (*ls == 0) {
    *ls = (n-1) / ((*gs)-1);
    if (*ls > max_l)
      *ls = max_l;
  }
  return GA_NO_ERROR;
}

int GpuKernel_setarg(GpuKernel *k, unsigned int i, void *a) {
  return k->ops->kernel_setarg(k->k, i, a);
}

int GpuKernel_call(GpuKernel *k, unsigned int n,
                   const size_t *bs, const size_t *gs,
                   size_t shared, void **args) {
  return k->ops->kernel_call(k->k, n, bs, gs, shared, args);
}

int GpuKernel_binary(const GpuKernel *k, size_t *sz, void **bin) {
  return k->ops->kernel_binary(k->k, sz, bin);
}

const char *GpuKernel_error(const GpuKernel *k, int err) {
  void *ctx;
  int err2 = k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_CTX, &ctx);
  if (err2 != GA_NO_ERROR) {
    /* If CUDA refuses to work after any kind of error in kernels
       there is not much we can do about it. */
    return gpuarray_error_str(err);
  }
  return Gpu_error(k->ops, ctx, err);
}
