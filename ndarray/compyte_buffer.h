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
  int (*buffer_move)(gpudata *dst, gpudata *src, size_t sz);
  /* device to host */
  int (*buffer_read)(void *dst, gpudata *src, size_t sz);
  /* host to device */
  int (*buffer_write)(gpudata *dst, void *src, size_t sz);
  /* Set buffer to a single-byte pattern (like C memset) */
  int (*buffer_memset)(gpudata *dst, int data, size_t sz);
  /* Add the specified offset into the buffer, 
     must not go beyond the buffer limits */
  int (*buffer_offset)(gpudata *buf, int offset);

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
  int nd;
  int flags;
  
  /* XXX: Replace this by a dtype code like numpy? */
  size_t elsize;
  size_t *dimensions;
  ssize_t *strides;
  size_t total_size;
  /* Try to keep in sync with numpy values for now*/
#define GA_C_CONTIGUOUS   0x0001
#define GA_F_CONTIGUOUS   0x0002
#define GA_OWNDATA        0x0004
#define GA_ENSURECOPY     0x0020
#define GA_ALIGNED        0x0100
#define GA_WRITEABLE      0x0400
#define GA_BEHAVED        (GA_ALIGNED|GA_WRITEABLE)
#define GA_CARRAY         (GA_C_CONTIGUOUS|GA_BEHAVED)
#define GA_FARRAY         (GA_F_CONTIGUOUS|GA_BEHAVED)
  /* Numpy flags that will not be supported at this level (and why):

     NPY_NOTSWAPPED: data is alway native endian
     NPY_FORCECAST: no casts
     NPY_ENSUREARRAY: no inherited classes
     NPY_UPDATEIFCOPY: cannot support without refcount (or somesuch)

     Maybe will define other flags later */
} GpuArray;

typedef enum _ga_order {
  GA_ANY_ORDER=-1,
  GA_C_ORDER=0,
  GA_F_ORDER=1
} ga_order;

enum ga_error {
  GA_NO_ERROR = 0,
  GA_MEMORY_ERROR,
  GA_VALUE_ERROR,
  GA_IMPL_ERROR,
  GA_INVALID_ERROR,
  GA_UNSUPPORTED_ERROR,
  /* Add more error types if needed */
};

static inline int GpuArray_CHKFLAGS(GpuArray *a, int flags) {
  return a->flags & flags == flags;
}
/* Add tests here when you need them */
#define GpuArray_OWNSDATA(a) GpuArray_CHKFLAGS(a, GA_OWNDATA)
#define GpuArray_ISWRITEABLE(a) GpuArray_CHKFLAGS(a, GA_WRITEABLE)
#define GpuArray_ISALIGNED(a) GpuArray_CHKFLAGS(a, GA_ALIGNED)
#define GpuArray_ISONESEGMENT(a) ((a)->flags & (GA_C_CONTIGUOUS|GA_F_CONTIGUOUS))
#define GpuArray_ISFORTRAN(a) GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS)
#define GpuArray_ITEMSIZE(a) ((a)->elsize) /* For now */

int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops, void *ctx, int flags,
		   size_t elsize, int nd, size_t *dims, ga_order ord);
int GpuArray_zeros(GpuArray *a, compyte_buffer_ops *ops, void *ctx, int flags,
		   size_t elsize, int nd, size_t *dims, ga_order ord);

void GpuArray_clear(GpuArray *a);

int GpuArray_move(GpuArray *dst, GpuArray *src);
int GpuArray_write(GpuArray *dst, void *src, size_t src_sz);
int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src);

#ifdef __cplusplus
}
#endif

#endif
