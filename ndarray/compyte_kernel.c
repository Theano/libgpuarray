#include "compyte_kernel.h"
#include "compyte_error.h"

int GpuKernel_init(GpuKernel *k, compyte_buffer_ops *ops, void *ctx,
		   unsigned int count, const char **strs, size_t *lens,
		   const char *name, int flags) {
  int res = GA_NO_ERROR;

  k->ops = ops;
  k->k = k->ops->buffer_newkernel(ctx, count, strs, lens, name, flags, &res);
  return res;
}

void GpuKernel_clear(GpuKernel *k) {
  if (k->k)
    k->ops->buffer_freekernel(k->k);
  k->k = NULL;
  k->ops = NULL;
}

int GpuKernel_setarg(GpuKernel *k, unsigned int index, int typecode,
		     void *arg) {
  return k->ops->buffer_setkernelarg(k->k, index, typecode, arg);
}

int GpuKernel_setbufarg(GpuKernel *k, unsigned int index, GpuArray *a) {
  return k->ops->buffer_setkernelargbuf(k->k, index, a->data);
}

int GpuKernel_call(GpuKernel *k, size_t n) {
  return k->ops->buffer_callkernel(k->k, n);
}
