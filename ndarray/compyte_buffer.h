/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef COMPYTE_BUFFER_H
#define COMPYTE_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#include <GL/gl.h>

typedef void *gpubuf;

typedef struct _compyte_buffer_ops {
  /* This allocates a buffer of size sz on the device */
  gpubuf (*buffer_alloc)(size_t sz);
  void (*buffer_free)(gpubuf buf);
  
  /* device to device copy, no overlap */
  int (*buffer_move)(gpubuf dst, size_t dst_offset, 
		     gpubuf src, size_t src_offset, size_t sz);
  /* device to host */
  int (*buffer_read)(void *dst, gpubuf src, size_t src_offset, size_t sz);
  /* host to device */
  int (*buffer_write)(gpubuf dst, size_t dst_offset, void *src, size_t sz);
  /* Set buffer to a single-byte pattern (like C memset) */
  int (*buffer_memset)(gpubuf dst, int data, size_t sz);

  /* OpenGL interop to transfer data from one API to the other */
  /* Can be NULL, in which case inter-api transfers will not be allowed */
  GLuint (*buffer_to_opengl)(gpubuf b, size_t b_offset);
  gpubuf (*buffer_from_opengl)(GLuint b);

  /* Get a string describing the last error that happened */
  const char *(*buffer_error)(void);
} compyte_buffer_ops;

#ifdef WITH_CUDA
extern compyte_buffer_ops *cuda_ops;
#endif

#ifdef WITH_OPENCL
extern compyte_buffer_ops *opencl_ops;
#endif

#ifdef __cplusplus
}
#endif

#endif
