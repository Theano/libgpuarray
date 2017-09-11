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

struct _gpucontext;

/**
 * Opaque struct for context data.
 */
typedef struct _gpucontext gpucontext;

struct _gpukernel;

/**
 * Opaque struct for kernel data.
 */
typedef struct _gpukernel gpukernel;

/**
 * Gets information about the number of available platforms for the
 * backend specified in `name`.
 *
 * \param name the backend name
 * \param platcount will contain number of compatible
 *                  platforms in host
 *
 * \return #GA_NO_ERROR, if success
 */
GPUARRAY_PUBLIC int gpu_get_platform_count(const char* name,
                                           unsigned int* platcount);

/**
 * Gets information about the number of compatible devices on a
 * specific host's `platform` for the backend specified in `name`.
 *
 * \param name the backend name
 * \param platform number for a platform in host
 * \param devcount will contain number of compatible devices in
 *                 `platform`
 *
 * \return #GA_NO_ERROR, if success
 */
GPUARRAY_PUBLIC int gpu_get_device_count(const char* name,
                                         unsigned int platform,
                                         unsigned int* devcount);


/**
 * Opaque structure that holds properties for the context.
 */
typedef struct _gpucontext_props gpucontext_props;

/**
 * Allocate and initialized an instance of gpucontext_props.
 *
 * Initialization is done with default values.
 *
 * \param res pointer to storage space for the created object
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_new(gpucontext_props **res);

/**
 * Set the device number for a CUDA device.
 *
 * \param p properties object
 * \param devno device number
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_cuda_dev(gpucontext_props *p, int devno);


/**
 * Set the platform and device for OpenCL.
 *
 * \param p properties object
 * \param platno platform number
 * \param devno device number
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_opencl_dev(gpucontext_props *p,
                                                int platno, int devno);

/**
 * Set the scheduling mode for the device.
 *
 * \param p properties object
 * \param sched scheduling mode.  One of \ref sched_modes "these".
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_sched(gpucontext_props *p, int sched);

/** \defgroup sched_modes
 * @{
 */

/**
 * Automatic scheduling, decide what to do depending on the workload,
 * number of cores in the computer and other relevant factors. (default)
 */
#define GA_CTX_SCHED_AUTO   0

/**
 * Single-work scheduling.  Optimize for speed in a single process,
 * with a single thread.  This is the fastest mode, but it may keep
 * the CPU busy more than necessary.
 */
#define GA_CTX_SCHED_SINGLE 1

/**
 * Multi-work scheduling.  Try to not keep the CPU busy more than
 * necessary and let other threads a chance at some CPU time.  This
 * may increase the latency when waiting for GPU operations.
 */
#define GA_CTX_SCHED_MULTI  2

/** @}*/

/**
 * Set single-stream mode.
 *
 * All operations on the device will be serialized on a single stream.
 * This will also disable most of the interlocking normally done
 * between multiple streams to keep everything in order.
 *
 * This mode can be faster if you don't have a lot of device-level
 * parallelism in your workload.
 *
 * \param p properties object
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_set_single_stream(gpucontext_props *p);

/**
 * Set the path for the kernel cache.
 *
 * The cache can be shared with other running instances, even on
 * shared drives.
 *
 * \param p properties object
 * \param path desired location of the kernel cache
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_kernel_cache(gpucontext_props *p,
                                                  const char *path);

/**
 * Configure the allocation cache.
 *
 * The maximum size is also a limit on the total amount of memory
 * allocated on the device.
 *
 * \param p properties object
 * \param initial initial size of the cache
 * \param max maximum size of the cache
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_props_alloc_cache(gpucontext_props *p,
                                                 size_t initial, size_t max);

/**
 * Free a properties object.
 *
 * This should not be called on a properties object that has been
 * passed to gpucontext_init().
 *
 * \param p properties object
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC void gpucontext_props_del(gpucontext_props *p);

/**
 * Create a context on the specified device.
 *
 * \warning This function is not thread-safe.
 *
 * The passed-in properties pointer will be managed by this function
 * and needs not be freed.  This means that you shouldn't touch the
 * properties object after passing it to this function.
 *
 * \param res a pointer to a location that will be allocated
 * \param name the backend name.
 * \param props a properties object for the context.  Can be NULL for
 *              defaults.
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_init(gpucontext **res, const char *name,
                                    gpucontext_props *props);

/**
 * Dereference a context.
 *
 * This removes a reference to the context and as soon as the
 * reference count drops to zero the context is destroyed.  The
 * context can stay alive after you call this function because some
 * object keep a reference to their context.
 *
 * \param ctx a valid context pointer.
 */
GPUARRAY_PUBLIC void gpucontext_deref(gpucontext *ctx);

/**
 * Fetch a context property.
 *
 * The property must be a context property.  The currently defined
 * properties and their type are defined in \ref props "Properties".
 *
 * \param ctx context
 * \param prop_id property id (from \ref props "Properties")
 * \param res pointer to the return space of the appropriate type
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpucontext_property(gpucontext *ctx, int prop_id,
                                        void *res);

/**
 * Get a string describing `err`.
 *
 * If you need to get a description of a error that occurred during
 * context creation, call this function using NULL as the context.
 * This version of the call is not thread-safe.
 *
 * \param ctx the context in which the error occured
 * \param err error code
 *
 * \returns string description of error
 */
GPUARRAY_PUBLIC const char *gpucontext_error(gpucontext *ctx, int err);

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
GPUARRAY_PUBLIC gpudata *gpudata_alloc(gpucontext *ctx, size_t sz, void *data,
                                       int flags, int *ret);

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
 * You can use gpudata_write() to set the contents.
 *
 * You may not call gpudata_memset() with the resulting buffer as the
 * destination.
 */
#define GA_BUFFER_READ_ONLY  0x01

/**
 * Signal that the memory in this buffer will only be written by
 * kernels (i.e. it is an output buffer).
 *
 * You can read the contents with gpudata_read().
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

/* The upper 16 bits are private flags */
#define GA_BUFFER_MASK       0xffff

/**
 * @}
 */

/**
 * Increase the reference count to the passed buffer by 1.
 *
 * \param b a buffer
 */
GPUARRAY_PUBLIC void gpudata_retain(gpudata *b);

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
GPUARRAY_PUBLIC void gpudata_release(gpudata *b);

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
GPUARRAY_PUBLIC int gpudata_share(gpudata *a, gpudata *b, int *ret);

/**
 * Copy the content of a buffer to another.
 *
 * Both buffers must be in the same context and contiguous.
 * Additionally the buffers must not overlap otherwise the content of
 * the destination buffer is not defined.
 *
 * \param dst destination buffer
 * \param dstoff offset inside the destination buffer
 * \param src source buffer
 * \param srcoff offset inside the source buffer
 * \param sz size of data to copy (in bytes)
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpudata_move(gpudata *dst, size_t dstoff,
                                 gpudata *src, size_t srcoff,
                                 size_t sz);

/**
 * Transfer the content of buffer across contexts.
 *
 * If possible it will try to the the transfer in an efficient way
 * using backend-specific tricks.  If those fail or can't be used, it
 * will fallback to a copy through the host.
 *
 * \param dst buffer to transfer to
 * \param dstoff offset in the destination buffer
 * \param src buffer to transfer from
 * \param srcoff offset in the source buffer
 * \param sz size of the region to transfer
 *
 * \returns the new buffer in dst_ctx or NULL if no efficient way to
 *          transfer could be found.
 */
GPUARRAY_PUBLIC int gpudata_transfer(gpudata *dst, size_t dstoff,
                                     gpudata *src, size_t srcoff,
                                     size_t sz);

/**
 * Transfer data from a buffer to memory.
 *
 * The buffer and the memory region must be contiguous.
 *
 * \param dst destination in memory
 * \param src source buffer
 * \param srcoff offset inside the source buffer
 * \param sz size of data to copy (in bytes)
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpudata_read(void *dst,
                                 gpudata *src, size_t srcoff,
                                 size_t sz);

/**
 * Transfer data from memory to a buffer.
 *
 * The buffer and the memory region must be contiguous.
 *
 * \param dst destination buffer
 * \param dstoff offset inside the destination buffer
 * \param src source in memory
 * \param sz size of data to copy (in bytes)
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpudata_write(gpudata *dst, size_t dstoff,
                                  const void *src, size_t sz);

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
GPUARRAY_PUBLIC int gpudata_memset(gpudata *dst, size_t dstoff, int data);

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
GPUARRAY_PUBLIC int gpudata_sync(gpudata *b);

/**
 * Fetch a buffer property.
 *
 * Can be used for buffer properties and context properties.  Context
 * properties will fetch the value for the context associated with the
 * buffer.  The currently defined properties and their type are
 * defined in \ref props "Properties".
 *
 * \param buf buffer
 * \param prop_id property id (from \ref props "Properties")
 * \param res pointer to the return space of the appropriate type
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpudata_property(gpudata *buf, int prop_id, void *res);

GPUARRAY_PUBLIC gpucontext *gpudata_context(gpudata *b);

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
 * \param numargs number of kernel arguments
 * \param typecodes the type of each argument
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
GPUARRAY_PUBLIC gpukernel *gpukernel_init(gpucontext *ctx, unsigned int count,
                                          const char **strings, const size_t *lengths,
                                          const char *fname, unsigned int numargs,
                                          const int *typecodes, int flags, int *ret,
                                          char **err_str);

/**
 * Retain a kernel.
 *
 * Increase the reference count of the passed kernel by 1.
 *
 * \param k a kernel
 */
GPUARRAY_PUBLIC void gpukernel_retain(gpukernel *k);

/**
 * Release a kernel.
 *
 * Decrease the reference count of a kernel.  If it reaches 0, all
 * resources associated with `k` will be released.
 *
 * If the reference count of a kernel reaches 0 while it is running,
 * this call will block until completion.
 */
GPUARRAY_PUBLIC void gpukernel_release(gpukernel *k);

/**
 * Set kernel argument.
 *
 * Buffer arguments will not be retained and it is the
 * responsability of the caller to ensure that the value is still
 * valid whenever a call is made.
 *
 * \param k kernel
 * \param i argument index (starting at 0)
 * \param a pointer to argument
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpukernel_setarg(gpukernel *k, unsigned int i, void *a);

/**
 * Call a kernel.
 *
 * If args is NULL, it will be assumed that the arguments have
 * previously been set with kernel_setarg().
 *
 * \param k kernel
 * \param n number of dimensions of grid/block
 * \param gs grid sizes for this call (also known as global size)
 * \param ls block sizes for this call (also known as local size)
 * \param shared amount of dynamic shared memory to reserve
 * \param args table of pointers to each argument (optional).
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpukernel_call(gpukernel *k, unsigned int n,
                                   const size_t *gs, const size_t *ls,
                                   size_t shared, void **args);

/**
 * Fetch a property.
 *
 * Can be used for kernel and context properties. The context
 * properties will fetch the value for the context associated with the
 * kernel.  The currently defined properties and their type are
 * defined in \ref props "Properties".
 *
 * \param k kernel
 * \param prop_id property id (from \ref props "Properties")
 * \param res pointer to the return space of the appropriate type
 *
 * \returns GA_NO_ERROR or an error code if an error occurred.
 */
GPUARRAY_PUBLIC int gpukernel_property(gpukernel *k, int prop_id, void *res);

GPUARRAY_PUBLIC gpucontext *gpukernel_context(gpukernel *k);

/**
 * \defgroup props Properties
 * @{
 */
/* Start at 1 for GA_CTX_PROP_ */
/**
 * Get the device name for the context.
 *
 * Type: `char [256]`
 */
#define GA_CTX_PROP_DEVNAME  1

/* UNUSED: 2 */

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

/* UNUSED: 5 */

/* UNUSED: 6 */

/**
 * Get the compatibility ID for the binaries generated with this context.
 *
 * Those binaries should work with any context which has the same ID.
 *
 * Type: `const char *`
 */
#define GA_CTX_PROP_BIN_ID    7

/**
 * Get a pre-allocated 8 byte buffer for kernel ops.
 *
 * This buffer is initialized to 0 on allocation and must always be
 * returned to that state after using it.
 *
 * This only to avoid the overhead of an allocation when calling a
 * kernel that may error out. It does not preclude the need for
 * synchronization and transfers.
 *
 * Type: `gpudata *`
 */
#define GA_CTX_PROP_ERRBUF    8

/**
 * Get the total size of global memory on the device.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_TOTAL_GMEM 9

/**
 * Get the size of free global memory on the device.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_FREE_GMEM 10

/**
 * Get the status of native float16 support on the device.
 *
 * Type: `int`
 */
#define GA_CTX_PROP_NATIVE_FLOAT16 11

/**
 * Get the maximum global size for dimension 0.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXGSIZE0 12

/**
 * Get the maximum global size for dimension 1.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXGSIZE1 13

/**
 * Get the maximum global size for dimension 2.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXGSIZE2 14

/**
 * Get the maximum local size for dimension 0.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXLSIZE0 15

/**
 * Get the maximum local size for dimension 1.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXLSIZE1 16

/**
 * Get the maximum loca size for dimension 2.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_MAXLSIZE2 17

/* UNUSED: 18 */

/**
 * Get a unique ID for the device behind the context.
 *
 * Type: `char [16]`
 */
#define GA_CTX_PROP_UNIQUE_ID 19

/**
 * Get the largest single block of memory that can be allocted.
 *
 * Type: `size_t`
 */
#define GA_CTX_PROP_LARGEST_MEMBLOCK 20

/* Start at 512 for GA_BUFFER_PROP_ */
#define GA_BUFFER_PROP_START  512

/**
 * Get the context in which this buffer was allocated.
 *
 * Type: `gpucontext *`
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
#define GA_KERNEL_PROP_START     1024

/**
 * Get the context for which this kernel was compiled.
 *
 * Type: `gpucontext *`
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
 * Flags for gpukernel_init().
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
  /* UNUSED: 0x01 */
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
  /* If you add a new flag, don't forget to update both
     gpuarray_buffer_{cuda,opencl}.c with the implementation of your flag */
  /**
   * The kernel is made of CUDA code.
   */
  GA_USE_CUDA =     0x2000,
  /**
   * The kernel is made of OpenCL code.
   */
  GA_USE_OPENCL =   0x4000,
} ga_usefl;

#ifdef __cplusplus
}
#endif

#endif
