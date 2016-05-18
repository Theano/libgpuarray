#include "gpuarray/kernel.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

#include <stdlib.h>

int GpuKernel_init(GpuKernel *k, gpucontext *ctx, unsigned int count,
                   const char **strs, const size_t *lens, const char *name,
                   unsigned int argcount, const int *types, int flags,
                   char **err_str) {
  int res = GA_NO_ERROR;

  k->args = calloc(argcount, sizeof(void *));
  if (k->args == NULL)
    return GA_MEMORY_ERROR;
  k->k = gpukernel_init(ctx, count, strs, lens, name, argcount, types,
                        flags, &res, err_str);
  if (res != GA_NO_ERROR)
    GpuKernel_clear(k);
  return res;
}

void GpuKernel_clear(GpuKernel *k) {
  if (k->k)
    gpukernel_release(k->k);
  free(k->args);
  k->k = NULL;
  k->args = NULL;
}

gpucontext *GpuKernel_context(GpuKernel *k) {
  return gpukernel_context(k->k);
}

int GpuKernel_sched(GpuKernel *k, size_t n, size_t *ls, size_t *gs) {
  size_t min_l;
  size_t max_l;
  size_t target_l;
  size_t max_g;
  size_t target_g;
  unsigned int numprocs;
  int err;
  int want_ls = 0;

  err = gpukernel_property(k->k, GA_KERNEL_PROP_MAXLSIZE, &max_l);
  if (err != GA_NO_ERROR)
    return err;
  err = gpukernel_property(k->k, GA_KERNEL_PROP_PREFLSIZE, &min_l);
  if (err != GA_NO_ERROR)
    return err;
  err = gpukernel_property(k->k, GA_CTX_PROP_NUMPROCS, &numprocs);
  if (err != GA_NO_ERROR)
    return err;
  err = gpukernel_property(k->k, GA_CTX_PROP_MAXGSIZE, &max_g);
  if (err != GA_NO_ERROR)
    return err;

  /* Do something about these hardcoded values */
  target_g = numprocs * 32;
  if (target_g > max_g)
    target_g = max_g;
  target_l = 512;
  if (target_l > max_l)
    target_l = max_l;

  if (*ls == 0) {
    want_ls = 1;
    *ls = min_l;
  }

  if (*gs == 0) {
    *gs = ((n-1) / *ls) + 1;
    if (*gs > target_g)
      *gs = target_g;
  }

  if (want_ls && n > (*ls * *gs)) {
    /* The division and multiplication by min_l is to ensure we end up
     * with a multiple of min_l */
    *ls = ((n / min_l) / *gs) * min_l;
    if (*ls > target_l)
      *ls = target_l;
  }

  return GA_NO_ERROR;
}

int GpuKernel_setarg(GpuKernel *k, unsigned int i, void *a) {
  return gpukernel_setarg(k->k, i, a);
}

int GpuKernel_call(GpuKernel *k, unsigned int n,
                   const size_t *bs, const size_t *gs,
                   size_t shared, void **args) {
  return gpukernel_call(k->k, n, bs, gs, shared, args);
}

int GpuKernel_binary(const GpuKernel *k, size_t *sz, void **bin) {
  return gpukernel_binary(k->k, sz, bin);
}

const char *GpuKernel_error(const GpuKernel *k, int err) {
  return gpucontext_error(gpukernel_context(k->k), err);
}
