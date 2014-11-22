/** \file buffer.h
 * \brief This file contains the interface definition for the backends.
 *
 * For normal use you should not call the functions defined in this
 * file directly.
 *
 * \see array.h For managing buffers
 * \see kernel.h For using kernels
 */
#ifndef GPUARRAY_BUFFER_H
#define GPUARRAY_BUFFER_H

#include <sys/types.h>
#include <stdio.h>
#include <stdarg.h>

#include <gpuarray/config.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

struct _gpudata;

/**
 * Opaque struct for buffer data.
 */
typedef struct _gpudata gpudata;

struct _gpukernel;

/**
 * Opaque struct for kernel data.
 */
typedef struct _gpukernel gpukernel;

/**
 * Function table that a backend must provide.
 * \headerfile gpuarray/buffer.h
 */
typedef struct _gpuarray_buffer_ops {
  /**
   * Create a context on the specified device.
   *
   * \warning This function is not thread-safe.
   *
   * \param dev the device number.  The precise meaning of the device
   *            number is backend-dependent
   * \param flags see \ref context_flags "Context flags"
   * \param ret error return location.  Will be ignored if set to NULL.
   *
   * \returns An opaque pointer to the created context or NULL if an
   * error occured.
   */
  void *(*buffer_init)(int dev, int flags, int *ret);

/**
 * \defgroup context_flags Context flags
 * @{
 */

/**
 * Let the backend decide on optimal parameters, using backend-defined
 * heuristics and defaults.
 *
 * This is the default (0) value.
 */
#define GA_CTX_DEFAULT       0x0

/**
 * Optimize parameters for multi-thread performance.
 *
 * May decrease overall performance in single-thread scenarios.
 */
#define GA_CTX_MULTI_THREAD  0x1

/**
 * Optimize parameters for single-thread performance.
 *
 * May decrease overall performace in multithread scenarios.
 */
#define GA_CTX_SINGLE_THREAD 0x2

/**
 * @}
 */

  /**
   * Destroy a context.
   *
   * This removes the external reference to the context and as soon as
   * all buffer and kernels associated with it are free all its
   * resources will be released.
   *
   * Do not call this function more than once on a given context.
   *
   * \param ctx a valid context pointer.
   */
  void (*buffer_deinit)(void *ctx);

  /**
   * Allocates a buffer of size `sz` in context `ctx`.
   *
   * Buffers are reference counted internally and start with a
   * reference count of 1.
   *
   * \param ctx a context pointer
   * \param sz the requested size
   * \param flags see \ref alloc_flags "Allocation flags"
   * \param data optional pointer to host buffer
   * \param ret error return pointer
   *
   * \returns A non-NULL pointer to a gpudata structure.  This
   * structure is intentionally opaque as its content may change
   * according to the backend used.
   */
  gpudata *(*buffer_alloc)(void *ctx, size_t sz, void *data, int flags,
                           int *ret);

/**
 * \defgroup alloc_flags Allocation flags
 * @{
 */

/**
 * The buffer is available for reading and writing from kernels.
 *
 * This is the default (0) value.
 */
#define GA_BUFFER_READ_WRITE 0x00

/**
 * Allocate the buffer in device-only memory.
 *
 * This is the default (0) value.
 */
#define GA_BUFFER_DEV        0x00

/**
 * Signal that the memory in this buffer will only be read by kernels.
 *
 * You can use buffer_write() to set the contents.
 *
 * You may not call buffer_extcopy() or buffer_memset() with the
 * resulting buffer as the destination.
 */
#define GA_BUFFER_READ_ONLY  0x01

/**
 * Signal that the memory in this buffer will only be written by
 * kernels (i.e. it is an output buffer).
 *
 * You can read the contents with buffer_read().
 *
 * You may not call buffer_extcopy() with the resuting buffer as
 * source.
 */

#define GA_BUFFER_WRITE_ONLY 0x02

/**
 * Initialize the contents of the buffer with the user-supplied host
 * buffer (`data`).  This buffer must be at least `sz` large.
 */
#define GA_BUFFER_INIT       0x04

/**
 * Allocate the buffer in host-reachable memory enabling you to
 * retrieve a pointer to the contents as the
 * `GA_BUFFER_PROP_HOSTPOINTER` property.
 */
#define GA_BUFFER_HOST       0x08

/*#define GA_BUFFER_USE_DATA   0x10*/

/**
 * @}
 */

  /**
   * Increase the reference count to the passed buffer by 1.
   *
   * \param b a buffer
   */
  void (*buffer_retain)(gpudata *b);

  /**
   * Release a buffer.
   *
   * This will decrement the reference count of the buffer by 1.  If
   * that count reaches 0 all associated ressources will be released.
   *
   * Even if your application does not have any references left to a
   * buffer it may still hang around if it is in use by internal
   * mechanisms (kernel call, ...)
   */
  void (*buffer_release)(gpudata *b);

  /**
   * Check if two buffers may overlap.
   *
   * Both buffers must have been created with the same backend.
   *
   * \param a first buffer
   * \param b second buffer
   * \param ret error return pointer
   *
   * \retval 1 The buffers may overlap
   * \retval 0 The buffers do not overlap.
   * \retval -1 An error was encoutered, `ret` contains a detailed
   * error code if not NULL.
   */
  int (*buffer_share)(gpudata *a, gpudata *b, int *ret);

  /**
   * Copy the content of a buffer to another.
   *
   * Both buffers must be in the same context.  Additionally the
   * buffers must not overlap otherwise the content of the destination
   * buffer is not defined.
   *
   * \param dst destination buffer
   * \param dstoff offset inside the destination buffer
   * \param src source buffer
   * \param srcoff offset inside the source buffer
   * \param sz size of data to copy (in bytes)
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_move)(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                     size_t sz);

  /**
   * Transfer data from a buffer to memory.
   *
   * \param dst destination in memory
   * \param src source buffer
   * \param srcoff offset inside the source buffer
   * \param sz size of data to copy (in bytes)
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_read)(void *dst, gpudata *src, size_t srcoff, size_t sz);

  /**
   * Transfer data from memory to a buffer.
   *
   * \param dst destination buffer
   * \param dstoff offset inside the destination buffer
   * \param src source in memory
   * \param sz size of data to copy (in bytes)
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_write)(gpudata *dst, size_t dstoff, const void *src, size_t sz);

  /**
   * Set a buffer to a byte pattern.
   *
   * This function acts like the C function memset() for device buffers.
   *
   * \param dst destination buffer
   * \param dstoff offset into the destination buffer
   * \param data byte value to write into the destination.
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_memset)(gpudata *dst, size_t dstoff, int data);

  /**
   * Compile a kernel.
   *
   * Compile the kernel composed of the concatenated strings in
   * `strings` and return a callable kernel.  If lengths is NULL then
   * all the strings must be NUL-terminated.  Otherwise, it doesn't
   * matter (but the lengths must not include the final NUL byte if
   * provided).
   *
   * \param ctx context to work in
   * \param count number of input strings
   * \param strings table of string pointers
   * \param lengths (optional) length for each string in the table
   * \param fname name of the kernel function (as defined in the code)
   * \param flags flags for compilation (see #ga_usefl)
   * \param ret error return pointer
   * \param err_str returns pointer to debug message from GPU backend 
   *        (if provided a non-NULL err_str)
   *
   * If `*err_str` is not NULL on return, the caller must call 
   * `free(*err_str)` after use.
   *
   * \returns Allocated kernel structure or NULL if an error occured.
   * `ret` will be updated with the error code if not NULL.
   */
  gpukernel *(*kernel_alloc)(void *ctx, unsigned int count,
                             const char **strings, const size_t *lengths,
                             const char *fname, unsigned int numargs,
                             const int *typecodes, int flags, int *ret, char **err_str);

  /**
   * Retain a kernel.
   *
   * Increase the reference count of the passed kernel by 1.
   *
   * \param k a kernel
   */
  void (*kernel_retain)(gpukernel *);

  /**
   * Release a kernel.
   *
   * Decrease the reference count of a kernel.  If it reaches 0, all
   * resources associated with `k` will be released.
   *
   * If the reference count of a kernel reaches 0 while it is running,
   * this call will block until completion.
   */
  void (*kernel_release)(gpukernel *k);

  /**
   * Call a kernel.
   *
   * \param k kernel
   * \param bs block size for this call (also known as local size)
   * \param gs grid size for this call (also known as global size)
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*kernel_call)(gpukernel *k, size_t bs[2], size_t gs[2], void **args);

  /**
   * Get the kernel binary.
   *
   * This can be use to cache kernel binaries after compilation of a
   * specific device.  The kernel can be recreated by calling
   * kernel_alloc with the binary and size and passing `GA_USE_BINARY`
   * as the use flags.
   *
   * The returned pointer is allocated and must be freed by the caller.
   *
   * \param k kernel
   * \param sz size of the returned binary
   * \param obj pointer to the binary for the kernel.
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*kernel_binary)(gpukernel *k, size_t *sz, void **obj);

  /**
   * Synchronize a buffer.
   *
   * Waits for all previous read, writes, copies and kernel calls
   * involving this buffer to be finished.
   *
   * This call is not required for normal use of the library as all
   * exposed operations will properly synchronize amongst themselves.
   * This call may be useful in a performance timing context to ensure
   * that the work is really done, or before interaction with another
   * library to wait for pending operations.
   */
  int (*buffer_sync)(gpudata *b);

  /**
   * Copy non-contiguous buffers.
   *
   * Specialized kernel to copy memory from a generic array structure
   * to another.  May be used to perform casts on the data and/or
   * change data layout.
   *
   * This function requires that the input and output buffers have the
   * same number of items.
   *
   * \param input input data buffer
   * \param ioff offset into input buffer
   * \param output output data buffer
   * \param ooff offset into output buffer
   * \param intype data type of input
   * \param outtype data type of output
   * \param a_nd order of input (number of dimensions)
   * \param a_dims dimensions of input
   * \param a_str strides of input
   * \param b_nd order of output (number of dimensions)
   * \param b_dims dimensions of output
   * \param b_str strides of output
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_extcopy)(gpudata *input, size_t ioff,
                        gpudata *output, size_t ooff,
                        int intype, int outtype,
                        unsigned int a_nd, const size_t *a_dims,
                        const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str);

  /**
   * Efficiently transfer the contents of a buffer to a new context.
   *
   * Both the buffer and the new context must come from the same
   * backend.
   *
   * If there is no way to do an efficient transfer within the
   * provided constraints, then this function may return NULL
   * indicating to the caller that it should fallback to some other
   * means of performing the transfer.
   *
   * \param src buffer to transfer from
   * \param offset start of the region transfer
   * \param sz size of the region to transfer
   * \param dst_ctx context for the result buffer
   * \param may_share set to 1 to allow the returned buffer to be a view
   *
   * \returns the new buffer in dst_ctx or NULL if no efficient way to
   *          transfer could be found.
   */
  gpudata *(*buffer_transfer)(gpudata *src, size_t offset, size_t sz,
                              void *dst_ctx, int may_share);

  /**
   * Fetch a property.
   *
   * Can be used to get a property of a context, a buffer or a kernel.
   * The element corresponding to the property category must be given
   * as argument and the other two are ignored.  The currently defined
   * properties and their type are defined in \ref props "Properties".
   *
   * \param ctx context
   * \param buf buffer
   * \param k kernel
   * \param prop_id property id (from \ref props "Properties")
   * \param res pointer to the return space of the appropriate type
   *
   * \returns GA_NO_ERROR or an error code if an error occurred.
   */
  int (*property)(void *ctx, gpudata *buf, gpukernel *k, int prop_id,
                  void *res);

  /**
   * Get a string describing the last error that happened.
   *
   * This function will return a string description of the last
   * backend error to happen on the specified context.
   *
   * If you need to get a description of a error that occurred during
   * context creation, call this function using NULL as the context.
   * This version of the call is not thread-safe.
   *
   * \param ctx context for which to query the error
   *
   * \returns string description of the last error
   */
  const char *(*ctx_error)(void *ctx);
} gpuarray_buffer_ops;

/**
 * \defgroup props Properties
 * @{
 */
/* Start at 1 for GA_CTX_PROP_ */
/**
 * Get the device name for the context.
 *
 * \note The returned string is allocated and must be freed by the caller.
 *
 * Type: `char *`
 */
#define GA_CTX_PROP_DEVNAME  1

/**
 * Get the maximum block size (also known as local size) for a kernel
 * call in the context.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXLSIZE 2

/**
 * Get the local memory size available for a call in the context.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_LMEMSIZE 3

/**
 * Number of compute units in this context.
 *
 * compute units times local size is more or less the expected
 * parallelism available on the device, but this is a very rough
 * estimate.
 *
 * Type: `unsigned int`
 */
#define GA_CTX_PROP_NUMPROCS 4

/**
 * Get the maximum group size for a kernel call in this context.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXGSIZE  5

/**
 * Get the vector of blas ops for the context.
 *
 * This may differ from one context to the other in the same backend
 * depending of the availability and performance of various BLAS
 * libraries.
 *
 * Type: `const gpuarray_blas_ops *`
 */
#define GA_CTX_PROP_BLAS_OPS  6

/**
 * Get the compatibility ID for the binaries generated with this context.
 *
 * Those binaries should work with any context which has the same ID.
 *
 * Type: `const char *`
 */
#define GA_CTX_PROP_BIN_ID    7

/* Start at 512 for GA_BUFFER_PROP_ */
/**
 * Get the context in which this buffer was allocated.
 *
 * Type: `void *`
 */
#define GA_BUFFER_PROP_CTX    512

/**
 * The reference count of the buffer.  Use only for debugging purposes.
 *
 * Type: `unsigned int`
 */
#define GA_BUFFER_PROP_REFCNT 513

/**
 * Size of the buffer on the device.
 *
 * This may be larger than the requested allocation size due to a
 * number of factors.
 *
 * Type: `size_t`
 */
#define GA_BUFFER_PROP_SIZE  514

/* Start at 1024 for GA_KERNEL_PROP_ */
/**
 * Get the context for which this kernel was compiled.
 *
 * Type: `void *`
 */
#define GA_KERNEL_PROP_CTX       1024

/**
 * Get the maximum block size (also known as local size) for a call of
 * this kernel.
 *
 * Type: `size_t`
 */
#define GA_KERNEL_PROP_MAXLSIZE  1025

/**
 * Get the prefered multiple of the block size for a call to this
 * kernel.
 *
 * Type: `size_t`
 */
#define GA_KERNEL_PROP_PREFLSIZE 1026

/**
 * Get the number of kernel arguments.
 *
 * Type `unsigned int`
 */
#define GA_KERNEL_PROP_NUMARGS   1027

/**
 * Get the list of argument types for a kernel.
 *
 * This list is the same length as the number of arguments to the
 * kernel. Do not modify the returned list.
 *
 * Type: `const int *`
 */
#define GA_KERNEL_PROP_TYPES     1028

/**
 * @}
 */

/**
 * Flags for gpuarray_buffer_ops#buffer_newkernel.
 *
 * It is important to specify these properly as the compilation
 * machinery will ensure that the proper configuration is made to
 * support the requested features or error out if the demands cannot
 * be met.
 *
 * \warning Failure to properly specify the feature flags will in most
 * cases result in silent data corruption (especially on ATI cards).
 */
typedef enum _ga_usefl {
  /**
   * The kernel source uses CLUDA unified language.
   */
  GA_USE_CLUDA =      0x01,
  /**
   * The kernel makes use of small (size is smaller than 4 bytes) types.
   */
  GA_USE_SMALL =      0x02,
  /**
   * The kernel makes use of double or complex doubles.
   */
  GA_USE_DOUBLE =     0x04,
  /**
   * The kernel makes use of complex of complex doubles.
   */
  GA_USE_COMPLEX =    0x08,
  /**
   * The kernel makes use of half-floats (also known as float16)
   */
  GA_USE_HALF =       0x10,
  /**
   * The source code passed is actually a kernel binary.
   */
  GA_USE_BINARY =     0x20,
  /* If you add a new flag, don't forget to update both
     gpuarray_buffer_{cuda,opencl}.c with the implementation of your flag */
  /**
   * The kernel is made of PTX code.
   */
  GA_USE_PTX =      0x1000,
  /**
   * The kernel is made of CUDA code.
   */
  GA_USE_CUDA =     0x2000,
  /**
   * The kernel is make of OpenCL code.
   */
  GA_USE_OPENCL =   0x4000,
} ga_usefl;

/**
 * Get the error string corresponding to `err`.
 *
 * \param o operations vector for the backend that produced the error.
 * \param ctx the context in which the error occured or NULL if the
 *            error occured outside a context (like during context
 *            creation).
 * \param err error code
 *
 * \returns A string description of the error.
 */
GPUARRAY_PUBLIC const char *Gpu_error(const gpuarray_buffer_ops *o, void *ctx,
				     int err);
/**
 * Get operations vector for a backend.
 *
 * The available backends depend on how the library was built.
 *
 * \param name backend name, currently one of `"cuda"`  or `"opencl"`
 *
 * \returns the operation vector or NULL if the backend name is unrecognized.
 */
GPUARRAY_PUBLIC const gpuarray_buffer_ops *gpuarray_get_ops(const char *name);

/**
 * Transfer a buffer from one context to another.
 *
 * This function will try to use an efficient method if one is
 * available and fallback to going through host memory if nothing else
 * can be done.
 */
GPUARRAY_PUBLIC gpudata *gpuarray_buffer_transfer(gpudata *buf, size_t offset,
                                             size_t sz, void *src_ctx,
                                             const gpuarray_buffer_ops *src_ops,
                                             void *dst_ctx,
                                             const gpuarray_buffer_ops *dst_ops,
                                             int may_share, int *ret);

#ifdef __cplusplus
}
#endif

#endif
