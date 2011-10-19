/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef COMPYTE_BUFFER_H
#define COMPYTE_BUFFER_H

#include <sys/types.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

struct _gpudata;
typedef struct _gpudata gpudata;

typedef struct _compyte_buffer_ops {
  /* This allocates a buffer of size sz in context ctx */
  gpudata *(*buffer_alloc)(void *ctx, size_t sz);
  void (*buffer_free)(gpudata *);
  
  /* device to device copy, no overlap */
  int (*buffer_move)(gpudata *dst, size_t dst_offset, 
		     gpudata *src, size_t src_offset, size_t sz);
  /* device to host */
  int (*buffer_read)(void *dst, gpudata *src, size_t src_offset, size_t sz);
  /* host to device */
  int (*buffer_write)(gpudata *dst, size_t dst_offset, void *src, size_t sz);
  /* Set buffer to a single-byte pattern (like C memset) */
  int (*buffer_memset)(gpudata *dst, int data, size_t sz);

  /* Get a string describing the last error that happened 
     (may change if you make other api calls) */
  const char *(*buffer_error)(void);
} compyte_buffer_ops;

#ifdef WITH_CUDA
extern compyte_buffer_ops cuda_ops;
#endif

#ifdef WITH_OPENCL
extern compyte_buffer_ops opencl_ops;
#endif

typedef struct _GpuArray {
  gpudata *data;
  compyte_buffer_ops *ops;
  int offset;
  int nd;
  
  size_t elsize;
  size_t *dimensions;
  size_t *strides;
  int flags;
  /* Try to keep in sync with numpy values for now*/
#define GA_CONTIGUOUS     0x0001
#define GA_F_CONTIGUOUS   0x0002
#define GA_OWNDATA        0x0004
#define GA_ALIGNED        0x0100
#define GA_WRITEABLE      0x0400
#define GA_BEHAVED        (GA_ALIGNED|GA_WRITABLE)
#define GA_CARRAY         (GA_CONTIGUOUS|GA_BEHAVED)
#define GA_DEFAULT        GA_CARRAY
  /* Numpy flags that will not be supported at this level (and why):

     NPY_NOTSWAPPED: data is alway native endian
     NPY_FORCECAST: no casts
     NPY_ENSUREARRAY: no view functions or the like
     NPY_ENSURECOPY: no view functions or the like
     NPY_UPDATEIFCOPY: cannot support without refcount

     Maybe will define other flags later */
} GpuArray;

static inline int GpuArray_CHKFLAGS(GpuArray *a, int flags) {
  return a->flags & flags == flags;
}
/* Add tests here when you need them */
#define GpuArray_OWNSDATA(a) GpuArray_CHKFLAGS((a), GA_OWNDATA)

GpuArray *GpuArray_empty(compyte_buffer_ops *ops, void *ctx, int flags,
			 size_t elsize, int nd, size_t *dims);
GpuArray *GpuArray_zeros(void *ctx, int flags, size_t elsize, int nd, size_t *dims);

void GpuArray_free(GpuArray *a);

int GpuArray_move(GpuArray *dst, GpuArray *src);
int GpuArray_write(GpuArray *dst, void *src, size_t src_sz);
int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src);

int GpuArray_view(GpuArray *dst, GpuArray *a, int nd, size_t dims);

#ifdef __cplusplus
}
#endif

#endif
