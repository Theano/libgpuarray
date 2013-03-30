#ifndef COMPYTE_KERNEL_H
#define COMPYTE_KERNEL_H
/** \file kernel.h
 *  \brief Kernel functions.
 */

#include <compyte/buffer.h>
#include <compyte/array.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

/**
 * Kernel information structure.
 */
typedef struct _GpuKernel {
  /**
   * Device kernel reference.
   */
  gpukernel *k;
  /**
   * Backend operations vector.
   */
  compyte_buffer_ops *ops;
} GpuKernel;

COMPYTE_PUBLIC int GpuKernel_init(GpuKernel *, compyte_buffer_ops *ops,
                                  void *ctx, unsigned int count,
                                  const char **strs, size_t *lens,
                                  const char *name, int flags);

COMPYTE_PUBLIC void GpuKernel_clear(GpuKernel *);

COMPYTE_PUBLIC void *GpuKernel_context(GpuKernel *);

COMPYTE_PUBLIC int GpuKernel_setarg(GpuKernel *, unsigned int index,
                                    int typecode, void *arg);
COMPYTE_PUBLIC int GpuKernel_setbufarg(GpuKernel *, unsigned int index,
                                       GpuArray *);

COMPYTE_PUBLIC int GpuKernel_call(GpuKernel *, size_t n, size_t ls, size_t gs);

#ifdef __cplusplus
}
#endif

#endif
