/** \file buffer.h
 * This file contain the header for ALL code that depends on cuda or opencl.
 */
#ifndef COMPYTE_BUFFER_H
#define COMPYTE_BUFFER_H

#include <sys/types.h>
#include <stdio.h>
#include <stdarg.h>

#include <compyte/compat.h>

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

typedef struct _compyte_buffer_ops {
  /**
   * Create a context on the specified device.
   *
   * ### Parameters
   * - `dev` the device number.  The precise meaning of the device
   *   number is backend-dependent
   * - `ret` error return location.  Will be ignored if set to NULL.
   *
   * ### Return Value
   * An opaque pointer to the created context or NULL if an error occured.
   */
  void *(*buffer_init)(int dev, int *ret);
  /**
   * Destroy a context.
   *
   * This removes the external reference to the context and as soon as
   * all buffer and kernels associated with it are free all its
   * resources will be released.
   *
   * Do not call this function more than once on a given context.
   *
   * ### Parameters
   * - `ctx` a valid context pointer.
   */
  void (*buffer_deinit)(void *ctx);
  /**
   * Allocates a buffer of size `sz` in context `ctx`.
   *
   * ### Parameters
   * - `ctx` a context pointer
   * - `sz` the requested size
   * - `ret` error return pointer
   *
   * ### Return Value
   * A non-NULL pointer to a gpudata structure.  This structure is
   * intentionally opaque as its content may change according to the
   * backend used.
   */
  gpudata *(*buffer_alloc)(void *ctx, size_t sz, int *ret);
  /**
   * Free a buffer.
   *
   * Release all ressources associated with `b`.
   *
   * If this function is called on a buffer that is in use by a kernel
   * the results are undefined.  (The current backend either block
   * until kernel completion or maintain a reference to the buffer,
   * but you should not rely on this behavior.)
   */
  void (*buffer_free)(gpudata *b);
  /**
   * Check if two buffers may overlap.
   *
   * ### Return Value
   * Return 1 if the buffers may overlap and 0 otherwise.  If there is
   * an error during processing -1 is returned and `ret`is set to the
   * appropriate error code if not NULL.
   */
  int (*buffer_share)(gpudata *, gpudata *, int *ret);
  
  /**
   * Copy the content of a buffer to another.
   *
   * Both buffers must be in the same context.  Additionally the
   * buffer must not overlap otherwise the content of the destination
   * buffer is not defined.
   *
   * ### Parameters
   * - `dst` destination buffer
   * - `dstoff` offset inside the destination buffer
   * - `src` source buffer
   * - `srcoff` offset inside the source buffer
   * - `sz` size of data to copy (in bytes)
   *
   * ### Return Value
   * GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_move)(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                     size_t sz);
  /**
   * Transfer data from a buffer to memory.
   *
   * ### Parameters
   * - `dst` destination in memory
   * - `src` source buffer
   * - `srcoff` offset inside the source buffer
   * - `sz` size of data to copy (in bytes)
   *
   * ### Return Value
   * GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_read)(void *dst, gpudata *src, size_t srcoff, size_t sz);
  /**
   * Transfer data from memory to a buffer.
   *
   * ### Parameters
   * - `dst` destination buffer
   * - `dstoff` offset inside the destination buffer
   * - `src` source in memory
   * - `sz` size of data to copy (in bytes)
   *
   * ### Return Value
   * GA_NO_ERROR or an error code if an error occurred.
   */
  int (*buffer_write)(gpudata *dst, size_t dstoff, const void *src, size_t sz);
  /* Set buffer to a single-byte pattern (like C memset) */
  int (*buffer_memset)(gpudata *dst, size_t dstoff, int data);
  /* Compile the kernel composed of the concatenated strings and return
     a callable kernel.  If lengths is NULL then all the strings must 
     be NUL-terminated.  Otherwise, it doesn't matter. */
  gpukernel *(*buffer_newkernel)(void *ctx, unsigned int count, 
				 const char **strings, const size_t *lengths,
				 const char *fname, int flags, int *ret);
  /* Free the kernel and all associated memory (including argument buffers) */
  void (*buffer_freekernel)(gpukernel *k);

  /* Copy the passed value to a kernel argument buffer */
  int (*buffer_setkernelarg)(gpukernel *k, unsigned int index,
			     int typecode, const void *val);
  
  /* Call the kernel with the previously specified arguments
     (this is synchronous only for now, might make async later) */
  int (*buffer_callkernel)(gpukernel *k, size_t bs, size_t gs);
  /* Make sure all compute operations involving this buffer are finished. */
  int (*buffer_sync)(gpudata *b);

  /* Function to facilitate copy and cast operations*/
  int (*buffer_extcopy)(gpudata *input, size_t ioff, gpudata *output, size_t ooff,
                        int intype, int outtype, unsigned int a_nd,
                        const size_t *a_dims, const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str);

  int (*buffer_property)(void *ctx, gpudata *buf, gpukernel *k, int prop_id,
                         void *res);

  /* Get a string describing the last error that happened
     (may change if you make other api calls) */
  const char *(*buffer_error)(void *ctx);
} compyte_buffer_ops;

/* Start at 1 for GA_CTX_PROP_ */
#define GA_CTX_PROP_DEVNAME  1
#define GA_CTX_PROP_MAXLSIZE 2
#define GA_CTX_PROP_LMEMSIZE 3
#define GA_CTX_PROP_NUMPROCS 4

/* Start at 512 for GA_BUFFER_PROP_ */
#define GA_BUFFER_PROP_CTX 512

/* Start at 1024 for GA_KERNEL_PROP_ */
#define GA_KERNEL_PROP_CTX       1024
#define GA_KERNEL_PROP_MAXLSIZE  1025
#define GA_KERNEL_PROP_PREFLSIZE 1026
#define GA_KERNEL_PROP_MAXGSIZE  1027

typedef enum _ga_usefl {
  GA_USE_CLUDA =      0x01,
  GA_USE_SMALL =      0x02,
  GA_USE_DOUBLE =     0x04,
  GA_USE_COMPLEX =    0x08,
  GA_USE_HALF =       0x10,
  /* If you add a new flag, don't forget to update both
     compyte_buffer_{cuda,opencl}.c with the implementation of your flag */
  GA_USE_PTX =      0x1000,
} ga_usefl;

COMPYTE_PUBLIC const char *Gpu_error(compyte_buffer_ops *o, void *ctx,
				     int err);
COMPYTE_PUBLIC compyte_buffer_ops *compyte_get_ops(const char *name);

#ifdef __cplusplus
}
#endif

#endif
