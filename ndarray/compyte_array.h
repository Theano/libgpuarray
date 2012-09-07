#ifndef COMPYTE_ARRAY_H
#define COMPYTE_ARRAY_H

#include "compyte_buffer.h"

typedef struct _GpuArray {
  gpudata *data;
  compyte_buffer_ops *ops;
  size_t *dimensions;
  ssize_t *strides;
  size_t offset;
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

typedef enum _ga_order {
  GA_ANY_ORDER=-1,
  GA_C_ORDER=0,
  GA_F_ORDER=1
} ga_order;

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

COMPYTE_PUBLIC int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                   int typecode, unsigned int nd, size_t *dims, ga_order ord);
COMPYTE_PUBLIC int GpuArray_zeros(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                   int typecode, unsigned int nd, size_t *dims, ga_order ord);

COMPYTE_PUBLIC int GpuArray_fromdata(GpuArray *a, compyte_buffer_ops *ops, gpudata *data,
                      size_t offset, int typecode, unsigned int nd, size_t *dims,
                      ssize_t *strides, int writeable);

COMPYTE_PUBLIC int GpuArray_view(GpuArray *v, GpuArray *a);
COMPYTE_PUBLIC int GpuArray_index(GpuArray *r, GpuArray *a, ssize_t *starts, ssize_t *stops,
                   ssize_t *steps);

COMPYTE_PUBLIC void GpuArray_clear(GpuArray *a);

COMPYTE_PUBLIC int GpuArray_share(GpuArray *a, GpuArray *b);

COMPYTE_PUBLIC int GpuArray_move(GpuArray *dst, GpuArray *src);
COMPYTE_PUBLIC int GpuArray_write(GpuArray *dst, void *src, size_t src_sz);
COMPYTE_PUBLIC int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src);

COMPYTE_PUBLIC int GpuArray_memset(GpuArray *a, int data);

COMPYTE_PUBLIC const char *GpuArray_error(GpuArray *a, int err);

COMPYTE_PUBLIC void GpuArray_fprintf(FILE *fd, const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);
#endif
