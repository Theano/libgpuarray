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
 * \param ctx context in which to build the kernel
 * \param count number of source code strings
 * \param strs C array of source code strings
 * \param lens C array with the size of each string or NULL
 * \param name name of the kernel function
 * \param flags kernel use flags (see \ref ga_usefl)
 * \param err_str (if not NULL) location to write GPU-backend provided debug info 
 * 
 * If `*err_str` is returned not NULL then it must be free()d by the caller
 *
 * \return GA_NO_ERROR if the operation is successful
 * \return any other value if an error occured
 */
GPUARRAY_PUBLIC int GpuKernel_init(GpuKernel *k, gpucontext *ctx,
                                   unsigned int count, const char **strs,
                                   const size_t *lens, const char *name,
                                   unsigned int argcount, const int *types,
                                   int flags, char **err_str);

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
GPUARRAY_PUBLIC gpucontext *GpuKernel_context(GpuKernel *k);


GPUARRAY_PUBLIC int GpuKernel_setarg(GpuKernel *k, unsigned int i, void *val);

/**
 * Do a scheduling of local and global size for a kernel.
 *
 * This function will find an optimal grid and block size for the
 * number of elements specified in n when running kernel k.  The
 * parameters may run a bit more instances than n for efficiency
 * reasons, so your kernel must be ready to deal with that.
 *
 * If either gs or ls is not 0 on entry its value will not be altered
 * and will be taken into account when choosing the other value.
 *
 * \param k the kernel to schedule for
 * \param n number of elements to handle
 * \param ls local size (in/out)
 * \param gs grid size (in/out)
 */
GPUARRAY_PUBLIC int GpuKernel_sched(GpuKernel *k, size_t n,
                                    size_t *ls, size_t *gs);

/**
 * Launch the execution of a kernel.
 *
 * \param k the kernel to launch
 * \param n dimensionality of the grid/blocks
 * \param ls sizes of launch blocks
 * \param gs sizes of launch grid
 * \param amount of dynamic shared memory to allocate
 * \param args table of pointers to arguments
 */
GPUARRAY_PUBLIC int GpuKernel_call(GpuKernel *k, unsigned int n,
                                   const size_t *ls, const size_t *gs,
                                   size_t shared, void **args);

GPUARRAY_PUBLIC int GpuKernel_binary(const GpuKernel *k, size_t *sz,
                                    void **obj);

GPUARRAY_PUBLIC const char *GpuKernel_error(const GpuKernel *k, int err);

#ifdef __cplusplus
}
#endif

#endif
