/**
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
typedef struct _gpudata gpudata;

struct _gpukernel;
typedef struct _gpukernel gpukernel;

typedef struct _compyte_buffer_ops {
  /* This allocates a buffer of size sz in context ctx */
  void *(*buffer_init)(int dev, int *ret);
  void (*buffer_deinit)(void *ctx);
  gpudata *(*buffer_alloc)(void *ctx, size_t sz, int *ret);
  void (*buffer_free)(gpudata *);
  int (*buffer_share)(gpudata *, gpudata *, int *ret);
  
  /* device to device copy, no overlap */
  int (*buffer_move)(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff, size_t sz);
  /* device to host */
  int (*buffer_read)(void *dst, gpudata *src, size_t srcoff, size_t sz);
  /* host to device */
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
  int (*buffer_setkernelargbuf)(gpukernel *k, unsigned int index,
				gpudata *b);
  
  /* Call the kernel with the previously specified arguments
     (this is synchronous only for now, might make async later) */
  int (*buffer_callkernel)(gpukernel *k, size_t n);

  /* Function to facilitate copy and cast operations*/
  int (*buffer_extcopy)(gpudata *input, size_t ioff, gpudata *output, size_t ooff,
                        int intype, int outtype, unsigned int a_nd,
                        const size_t *a_dims, const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str);

  /* Get a string describing the last error that happened
     (may change if you make other api calls) */
  const char *(*buffer_error)(gpudata *b);
} compyte_buffer_ops;

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

COMPYTE_PUBLIC const char *Gpu_error(compyte_buffer_ops *o, gpudata *b,
				     int err);
COMPYTE_PUBLIC compyte_buffer_ops *compyte_get_ops(const char *name);

#ifdef __cplusplus
}
#endif

#endif
