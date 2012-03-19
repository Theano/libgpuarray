/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef COMPYTE_BUFFER_H
#define COMPYTE_BUFFER_H

#include <sys/types.h>
#include <stdio.h>

#include "compyte_util.h"

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
  gpudata *(*buffer_alloc)(void *ctx, size_t sz, int *ret);
  gpudata *(*buffer_dup)(gpudata *b, int *ret);
  void (*buffer_free)(gpudata *);
  int (*buffer_share)(gpudata *, gpudata *, int *ret);
  
  /* device to device copy, no overlap */
  int (*buffer_move)(gpudata *dst, gpudata *src);
  /* device to host */
  int (*buffer_read)(void *dst, gpudata *src, size_t sz);
  /* host to device */
  int (*buffer_write)(gpudata *dst, const void *src, size_t sz);
  /* Set buffer to a single-byte pattern (like C memset) */
  int (*buffer_memset)(gpudata *dst, int data);
  /* Add the specified offset into the buffer, 
     must not go beyond the buffer limits */
  int (*buffer_offset)(gpudata *buf, ssize_t offset);
  /* Compile the kernel composed of the concatenated strings and return
     a callable kernel.  If lengths is NULL then all the strings must 
     be NUL-terminated.  Otherwise, it doesn't matter. */
  gpukernel *(*buffer_newkernel)(void *ctx, unsigned int count, 
				 const char **strings, const size_t *lengths,
				 const char *fname, int *ret);
  /* Free the kernel and all associated memory (including argument buffers) */
  void (*buffer_freekernel)(gpukernel *k);
  
  /* Copy the passed value to a kernel argument buffer */
  int (*buffer_setkernelarg)(gpukernel *k, unsigned int index,
			     size_t sz, const void *val);
  int (*buffer_setkernelargbuf)(gpukernel *k, unsigned int index,
				gpudata *b);
  
  /* Call the kernel with the previously specified arguments
     (this is synchronous only for now, might make async later) */
  int (*buffer_callkernel)(gpukernel *k, unsigned int gx, unsigned int gy,
			   unsigned int gz, unsigned int lx, unsigned int ly,
			   unsigned int lz);

  /* Function to facilitate copy and cast operations*/
  int (*buffer_elemwise)(gpudata *input, gpudata *output, int intype,
			 int outtype, const char *op, unsigned int a_nd,
			 const size_t *a_dims, const ssize_t *a_str,
			 unsigned int b_nd, const size_t *b_dims,
			 const ssize_t *b_str);

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
  size_t *dimensions;
  ssize_t *strides;
  unsigned int nd;
  int flags;
  int typecode;

  /* Try to keep in sync with numpy values for now */
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

typedef struct _GpuKernel {
  gpukernel *k;
  compyte_buffer_ops *ops;
} GpuKernel;

typedef enum _ga_order {
  GA_ANY_ORDER=-1,
  GA_C_ORDER=0,
  GA_F_ORDER=1
} ga_order;

enum ga_error {
  GA_NO_ERROR = 0,
  GA_MEMORY_ERROR,
  GA_VALUE_ERROR,
  GA_IMPL_ERROR, /* call buffer_error() for more details */
  GA_INVALID_ERROR,
  GA_UNSUPPORTED_ERROR,
  GA_SYS_ERROR, /* look at errno for more details */
  GA_RUN_ERROR,
  /* Add more error types if needed */
  /* Don't forget to sync with Gpu_error() */
};

const char *Gpu_error(compyte_buffer_ops *o, int err);

static inline int GpuArray_CHKFLAGS(GpuArray *a, int flags) {
  return (a->flags & flags) == flags;
}
/* Add tests here when you need them */
#define GpuArray_OWNSDATA(a) GpuArray_CHKFLAGS(a, GA_OWNDATA)
#define GpuArray_ISWRITEABLE(a) GpuArray_CHKFLAGS(a, GA_WRITEABLE)
#define GpuArray_ISALIGNED(a) GpuArray_CHKFLAGS(a, GA_ALIGNED)
#define GpuArray_ISONESEGMENT(a) ((a)->flags & (GA_C_CONTIGUOUS|GA_F_CONTIGUOUS))
#define GpuArray_ISFORTRAN(a) GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS)
#define GpuArray_ITEMSIZE(a) compyte_get_elsize((a)->typecode)

int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, size_t *dims, ga_order ord);
int GpuArray_zeros(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, size_t *dims, ga_order ord);

int GpuArray_view(GpuArray *v, GpuArray *a);
int GpuArray_index(GpuArray *r, GpuArray *a, ssize_t *starts, ssize_t *stops,
		   ssize_t *steps);

void GpuArray_clear(GpuArray *a);

int GpuArray_share(GpuArray *a, GpuArray *b);

int GpuArray_move(GpuArray *dst, GpuArray *src);
int GpuArray_write(GpuArray *dst, void *src, size_t src_sz);
int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src);

int GpuArray_memset(GpuArray *a, int data);

const char *GpuArray_error(GpuArray *a, int err);

void GpuArray_fprintf(FILE *fd, const GpuArray *a);
int GpuArray_is_c_contiguous(const GpuArray *a);
int GpuArray_is_f_contiguous(const GpuArray *a);

int GpuKernel_init(GpuKernel *, compyte_buffer_ops *ops, void *ctx, int count,
		   const char **strs, size_t *lens, const char *name);

void GpuKernel_clear(GpuKernel *);

int GpuKernel_setarg(GpuKernel *, unsigned int index, int typecode, ...);
int GpuKernel_setbufarg(GpuKernel *, unsigned int index, GpuArray *);
int GpuKernel_setrawarg(GpuKernel *, unsigned int index, size_t sz, void *val);

int GpuKernel_call(GpuKernel *, unsigned int gx, unsigned int gy,
		   unsigned int gz, unsigned int lx, unsigned int ly,
		   unsigned int lz);

#ifdef __cplusplus
}
#endif

#endif
