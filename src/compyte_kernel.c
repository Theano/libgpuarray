#include "compyte/kernel.h"
#include "compyte/error.h"
#include "compyte/types.h"

int GpuKernel_init(GpuKernel *k, compyte_buffer_ops *ops, void *ctx,
		   unsigned int count, const char **strs, size_t *lens,
		   const char *name, int flags) {
  int res = GA_NO_ERROR;

  k->ops = ops;
  k->k = k->ops->buffer_newkernel(ctx, count, strs, lens, name, flags, &res);
  if (res != GA_NO_ERROR)
    GpuKernel_clear(k);
  return res;
}

void GpuKernel_clear(GpuKernel *k) {
  if (k->k)
    k->ops->buffer_freekernel(k->k);
  k->k = NULL;
  k->ops = NULL;
}

void *GpuKernel_context(GpuKernel *k) {
  void *res;
  (void)k->ops->buffer_property(NULL, NULL, k->k, GA_KERNEL_PROP_CTX, &res);
  return res;
}

int GpuKernel_setarg(GpuKernel *k, unsigned int index, int typecode,
		     void *arg) {
  return k->ops->buffer_setkernelarg(k->k, index, typecode, arg);
}

int GpuKernel_setbufarg(GpuKernel *k, unsigned int index, GpuArray *a) {
  return k->ops->buffer_setkernelarg(k->k, index, GA_BUFFER, a->data);
}

static int do_sched(GpuKernel *k, size_t n, size_t *ls, size_t *gs) {
  size_t min_l;
  size_t max_l;
  size_t max_g;
  int err;

  err = k->ops->buffer_property(NULL, NULL, k->k, GA_KERNEL_PROP_MAXLSIZE,
                                &max_l);
  if (err != GA_NO_ERROR)
    return err;
  err = k->ops->buffer_property(NULL, NULL, k->k, GA_KERNEL_PROP_PREFLSIZE,
                                &min_l);
  if (err != GA_NO_ERROR)
    return err;
  err = k->ops->buffer_property(NULL, NULL, k->k, GA_KERNEL_PROP_MAXGSIZE,
                                &max_g);
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

int GpuKernel_call(GpuKernel *k, size_t n, size_t bs, size_t gs) {
  int err;
  if (bs == 0 || gs == 0) {
    err = do_sched(k, n, &bs, &gs);
    if (err != GA_NO_ERROR)
      return err;
  }
  return k->ops->buffer_callkernel(k->k, bs, gs);
}
