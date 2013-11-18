#include "compyte/kernel.h"
#include "compyte/error.h"
#include "compyte/types.h"

int GpuKernel_init(GpuKernel *k, const compyte_buffer_ops *ops, void *ctx,
		   unsigned int count, const char **strs, size_t *lens,
		   const char *name, int flags) {
  int res = GA_NO_ERROR;

  k->ops = ops;
  k->k = k->ops->kernel_alloc(ctx, count, strs, lens, name, flags, &res);
  if (res != GA_NO_ERROR)
    GpuKernel_clear(k);
  return res;
}

void GpuKernel_clear(GpuKernel *k) {
  if (k->k)
    k->ops->kernel_release(k->k);
  k->k = NULL;
  k->ops = NULL;
}

void *GpuKernel_context(GpuKernel *k) {
  void *res;
  (void)k->ops->property(NULL, NULL, k->k, GA_KERNEL_PROP_CTX, &res);
  return res;
}

int GpuKernel_setarg(GpuKernel *k, unsigned int index, int typecode,
		     void *arg) {
  return k->ops->kernel_setarg(k->k, index, typecode, arg);
}

int GpuKernel_setbufarg(GpuKernel *k, unsigned int index, GpuArray *a) {
  if (!(a->flags | GA_ALIGNED))
    return GA_UNALIGNED_ERROR;
  return k->ops->kernel_setarg(k->k, index, GA_BUFFER, a->data);
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

int GpuKernel_call(GpuKernel *k, size_t n, size_t bs, size_t gs) {
  size_t _n[2], _bs[2], _gs[2];
  _n[1] = _bs[1] = _gs[1] = 1;
  _n[0] = n;
  _bs[0] = bs;
  _gs[0] = gs;
  return GpuKernel_call2(k, _n, _bs, _gs);
}

int GpuKernel_call2(GpuKernel *k, size_t n[2], size_t bs[2], size_t gs[2]) {
  size_t _bs[2] = {0, 0}, _gs[2] = {0, 0};
  int err;
  if (n == NULL) {
    if (bs == NULL || gs == NULL)
      return GA_INVALID_ERROR;
  } else {
    if (bs == NULL) bs = _bs;
    if (gs == NULL) gs = _gs;

    if (bs[0] == 0 || gs[0] == 0) {
      err = do_sched(k, n[0], &bs[0], &gs[0]);
      if (err != GA_NO_ERROR)
        return err;
    }

    if (bs[1] == 0 || gs[1] == 0) {
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
  return k->ops->kernel_call(k->k, bs, gs);
}
