#ifndef GPUARRAY_KERNEL_H
#define GPUARRAY_KERNEL_H
/** \file kernel.h
 *  \brief Kernel functions.
 */

#include <gpuarray/buffer.h>
#include <gpuarray/array.h>

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
  const gpuarray_buffer_ops *ops;
  /**
   * Argument buffer.
   */
  void **args;
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
GPUARRAY_PUBLIC int GpuKernel_init(GpuKernel *k, const gpuarray_buffer_ops *ops,
                                  void *ctx, unsigned int count,
                                  const char **strs, const size_t *lens,
                                  const char *name, unsigned int argcount,
                                  const int *types, int flags, char **err_str);

/**
 * Clear and release data associated with a kernel.
 *
 * \param k the kernel to release
 */
GPUARRAY_PUBLIC void GpuKernel_clear(GpuKernel *k);

/**
 * Returns the context in which a kernel was built.
 *
 * \param k a kernel
 *
 * \returns a context pointer
 */
GPUARRAY_PUBLIC void *GpuKernel_context(GpuKernel *k);

/**
 * Launch the execution of a kernel.
 *
 * You either specify the block and grid sizes (`ls` and `gs`) or the
 * total size (`n`). Set a value to `0` to indicate it is
 * unspecified. You can also specify the total size (`n`) and one of
 * the block (`ls`) or grid (`gs`) size.
 *
 * If you leave one or both of `ls` or `gs`, it will be filled
 * according to a heuristic to get a good performance out of your
 * hardware. However the number of kernel instances that will be run
 * can be slightly higher than the total size you specified in order
 * to avoid performance degradation. Your kernel should be ready to
 * handle this.
 *
 * \param k the kernel to launch
 * \param n number of instances to launch
 * \param ls size of launch blocks
 * \param gs size of launch grid
 */
GPUARRAY_PUBLIC int GpuKernel_call2(GpuKernel *k, size_t n[2],
                                   size_t ls[2], size_t gs[2], void **args);

GPUARRAY_PUBLIC int GpuKernel_call(GpuKernel *k, size_t n,
                                  size_t ls, size_t gs, void **args);

GPUARRAY_PUBLIC int GpuKernel_binary(const GpuKernel *k, size_t *sz,
                                    void **obj);

GPUARRAY_PUBLIC const char *GpuKernel_error(const GpuKernel *k, int err);

#ifdef __cplusplus
}
#endif

#endif
