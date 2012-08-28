#ifndef COMPYTE_KERNEL_H
#define COMPYTE_KERNEL_H

#include "compyte_buffer.h"
#include "compyte_array.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

typedef struct _GpuKernel {
  gpukernel *k;
  compyte_buffer_ops *ops;
} GpuKernel;

int GpuKernel_init(GpuKernel *, compyte_buffer_ops *ops, void *ctx,
                   unsigned int count, const char **strs, size_t *lens,
                   const char *name, int flags);

void GpuKernel_clear(GpuKernel *);

int GpuKernel_setarg(GpuKernel *, unsigned int index, int typecode, void *arg);
int GpuKernel_setbufarg(GpuKernel *, unsigned int index, GpuArray *);

int GpuKernel_call(GpuKernel *, size_t n);

#ifdef __cplusplus
}
#endif

#endif
