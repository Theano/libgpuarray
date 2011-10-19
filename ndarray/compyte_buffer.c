#include "compyte_buffer.h"

#include <strings.h>

void GpuArray_free(GpuArray *a) {
  if (res->data && GpuArray_OWNSDATA(a))
    res->ops->buffer_free(res->data);
  free(res->dimensions);
  free(res->strides);
  free(a);
}

#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

GpuArray *GpuArray_empty(compyte_buffer_ops *ops, void *ctx, int flags,
			 size_t elsize, int nd, size_t *dims) {
  size_t size = elsize;
  a->elsize = elsize;
  int i;

  for (i = 0; i < nd; i++) {
    size_t d = dims[i];
    if ((d >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
	d > 0 && SIZE_MAX / d < size)
      return NULL;
    size *= d;
  }
  GpuArray *res;
  res = malloc(sizeof(*res));
  if (res == NULL)
    return NULL;
  res->ops = ops;
  res->data = res->ops->buffer_alloc(ctx, size);
  res->offset = 0;
  res->nd = nd;
  res->elsize = elsize;
  res->dimensions = calloc(nd, sizeof(size_t));
  res->strides = calloc(nd, sizeof(size_t));
  res->flags = GA_DEFAULT|GA_OWNDATA;
  if (res->dimensions == NULL || res->strides == NULL || res->data == NULL) {
    GpuArray_free(res);
    return NULL;
  }
  /* Mult will not overflow since calloc succeded */
  bcopy(dims, res->dimensions, sizeof(size_t)*nd);
  for (i = 0; i < nd; i++) {
    /* XXX: do strides here */
  }
}
