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

static int do_sched(GpuKernel *k, size_t n, size_t *ls, size_t *gs) {
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

int GpuKernel_call(GpuKernel *k, size_t n, size_t bs, size_t gs, void **args) {
  size_t _n[2], _bs[2], _gs[2];
  _n[1] = _bs[1] = _gs[1] = 1;
  _n[0] = n;
  _bs[0] = bs;
  _gs[0] = gs;
  return GpuKernel_call2(k, _n, _bs, _gs, args);
}

int GpuKernel_call2(GpuKernel *k, size_t n[2], size_t _bs[2], size_t _gs[2],
                    void **args) {
  size_t bs[2] = {0, 0}, gs[2] = {0, 0};
  int *types;
  unsigned int argcount;
  unsigned int i;
  int err;

  if (_bs != NULL) bs[0] = _bs[0], bs[1] = _bs[1];
  if (_gs != NULL) gs[0] = _gs[0], gs[1] = _gs[1];
  if (n == NULL) {
    if (_bs == NULL || _gs == NULL ||
        bs[0] == 0 || bs[1] == 0 ||
        gs[0] == 0 || gs[1] == 0)
      return GA_INVALID_ERROR;
  } else {
    if (bs[0] == 0 || gs[0] == 0) {
      if (n[0] == 0)
        return GA_INVALID_ERROR;
      err = do_sched(k, n[0], &bs[0], &gs[0]);
      if (err != GA_NO_ERROR)
        return err;
    }

    if (bs[1] == 0 || gs[1] == 0) {
      if (n[1] == 0)
        return GA_INVALID_ERROR;
      if (n[1] == 1) {
        bs[1] = 1;
        gs[1] = 1;
      } else {
        err = do_sched(k, n[1], &bs[1], &gs[1]);
        if (err != GA_NO_ERROR)
          return err;
      }
    }
  }
  err = k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_NUMARGS, &argcount);
  if (err != GA_NO_ERROR) return err;
  err = k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_TYPES, &types);
  if (err != GA_NO_ERROR) return err;

  for (i = 0; i < argcount; i++)
    if (types[i] == GA_BUFFER)
      k->args[i] = ((GpuArray *)args[i])->data;
    else
      k->args[i] = args[i];
  return k->ops->kernel_call(k->k, bs, gs, k->args);
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
