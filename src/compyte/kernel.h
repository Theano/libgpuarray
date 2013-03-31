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

/**
 * Initialize a kernel structure.
 *
 * `lens` holds the size of each source string.  If is it NULL or an
 * element has a value of 0 the length will be determined using strlen()
 * or equivalent code.
 *
 * \param k a kernel structure
 * \param ops operations vector
 * \param ctx context in which to build the kernel
 * \param count number of source code strings
 * \param strs C array of source code strings
 * \param lens C array with the size of each string or NULL
 * \param name name of the kernel function
 * \param flags kernel use flags (see \ref ga_usefl)
 *
 * \return GA_NO_ERROR if the operation is successful
 * \return any other value if an error occured
 */
COMPYTE_PUBLIC int GpuKernel_init(GpuKernel *k, compyte_buffer_ops *ops,
                                  void *ctx, unsigned int count,
                                  const char **strs, size_t *lens,
                                  const char *name, int flags);

COMPYTE_PUBLIC void GpuKernel_clear(GpuKernel *k);

COMPYTE_PUBLIC void *GpuKernel_context(GpuKernel *k);

COMPYTE_PUBLIC int GpuKernel_setarg(GpuKernel *k, unsigned int index,
                                    int typecode, void *arg);
COMPYTE_PUBLIC int GpuKernel_setbufarg(GpuKernel *k, unsigned int index,
                                       GpuArray *a);

COMPYTE_PUBLIC int GpuKernel_call(GpuKernel *k, size_t n,
                                  size_t ls, size_t gs);

#ifdef __cplusplus
}
#endif

#endif
