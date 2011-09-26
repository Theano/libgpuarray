#ifndef COMPYTE_NDARRAY_H
#define COMPYTE_NDARRAY_H

#imclude "compyte_buffer.h"

typedef struct _compyte_ndarray {
  gpubuf data;
  size_t base_offset;
  unsigned int nd;
  ssize_t *dimensions;
  ssize_t *strides;
  
  int flags;
  pygpu_buffer_ops *ops;
} pygpu_ndarray;

#endif
