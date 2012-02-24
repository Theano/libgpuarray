#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <strings.h>
#include <errno.h>

#include "compyte_buffer.h"

#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, size_t *dims, ga_order ord) {
  size_t size = compyte_get_elsize(typecode);
  int i;

  if (ord == GA_ANY_ORDER)
    ord = GA_C_ORDER;

  if (ord != GA_C_ORDER && ord != GA_F_ORDER)
    return GA_VALUE_ERROR;

  for (i = 0; i < nd; i++) {
    size_t d = dims[i];
    if ((d >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
	d > 0 && SIZE_MAX / d < size)
      return GA_VALUE_ERROR;
    size *= d;
  }
  a->ops = ops;
  a->data = a->ops->buffer_alloc(ctx, size, NULL);
  a->nd = nd;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* F/C distinction comes later */
  a->flags = GA_OWNDATA|GA_BEHAVED;
  if (a->dimensions == NULL || a->strides == NULL || a->data == NULL) {
    GpuArray_clear(a);
    return GA_MEMORY_ERROR;
  }
  /* Mult will not overflow since calloc succeded */
  bcopy(dims, a->dimensions, sizeof(size_t)*nd);

  size = compyte_get_elsize(typecode);
  /* mults will not overflow, checked on entry */
  switch (ord) {
  case GA_C_ORDER:
    for (i = nd-1; i >= 0; i--) {
      a->strides[i] = size;
      size *= a->dimensions[i];
    }
    a->flags |= GA_C_CONTIGUOUS;
    break;
  case GA_F_ORDER:
    for (i = 0; i < nd; i++) {
      a->strides[i] = size;
      size *= a->dimensions[i];
    }
    a->flags |= GA_F_CONTIGUOUS;
    break;
  default:
    assert(0); /* cannot be reached */
  }

  if (a->nd <= 1)
    a->flags |= GA_F_CONTIGUOUS|GA_C_CONTIGUOUS;

  return GA_NO_ERROR;
}

int GpuArray_zeros(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                   int typecode, unsigned int nd, size_t *dims, ga_order ord) {
  int err;
  err = GpuArray_empty(a, ops, ctx, typecode, nd, dims, ord);
  if (err != GA_NO_ERROR)
    return err;
  err = a->ops->buffer_memset(a->data, 0);
  if (err != GA_NO_ERROR) {
    GpuArray_clear(a);
  }
  return err;
}

int GpuArray_view(GpuArray *v, GpuArray *a) {
  v->ops = a->ops;
  v->data = a->data;
  v->nd = a->nd;
  v->typecode = a->typecode;
  v->flags = a->flags & ~GA_OWNDATA;
  v->dimensions = calloc(v->nd, sizeof(size_t));
  v->strides = calloc(v->nd, sizeof(ssize_t));
  if (v->dimensions == NULL || v->strides == NULL) {
    GpuArray_clear(v);
    return GA_MEMORY_ERROR;
  }
  bcopy(a->dimensions, v->dimensions, v->nd*sizeof(size_t));
  bcopy(a->strides, v->strides, v->nd*sizeof(ssize_t));
  return GA_NO_ERROR;
}

int GpuArray_index(GpuArray *r, GpuArray *a, size_t *starts, size_t *stops,
		   ssize_t *steps) {
  int err;
  unsigned int i, r_i;
  unsigned int new_nd = a->nd;

  if ((starts == NULL) || (stops == NULL) || (steps == NULL))
    return GA_VALUE_ERROR;

  for (i = 0; i < r->nd; i++) {
    if (steps[i] == 0) new_nd -= 1;
  }

  r->ops = a->ops;
  r->data = a->ops->buffer_dup(a->data, &err);
  if (r->data == NULL) {
    GpuArray_clear(r);
    return err;
  }
  r->flags = a->flags;
  r->nd = new_nd;
  r->dimensions = calloc(r->nd, sizeof(size_t));
  r->strides = calloc(r->nd, sizeof(ssize_t));
  if (r->dimensions == NULL || r->strides == NULL) {
    GpuArray_clear(r);
    return GA_MEMORY_ERROR;
  }

  r_i = 0;
  for (i = 0; i < a->nd; i++) {
    if (starts[i] >= a->dimensions[i]) {
      GpuArray_clear(r);
      return GA_VALUE_ERROR;
    }
    r->ops->buffer_offset(r->data, starts[i] * a->strides[i]);
    if (steps[i] != 0) {
      r->strides[r_i] = steps[i] * a->strides[i];
      r->dimensions[r_i] = (stops[i]-starts[i]+steps[i]-
			    (steps[i] < 0? -1 : 1))/steps[i];
      r_i++;
    }
    assert(r_i <= r->nd);
  }
  if (GpuArray_is_c_contiguous(r))
    r->flags |= GA_C_CONTIGUOUS;
  else
    r->flags &= ~GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(r))
    r->flags |= GA_F_CONTIGUOUS;
  else
    r->flags &= ~GA_F_CONTIGUOUS;

  return GA_NO_ERROR;
}

void GpuArray_clear(GpuArray *a) {
  if (a->data && GpuArray_OWNSDATA(a))
    a->ops->buffer_free(a->data);
  free(a->dimensions);
  free(a->strides);
  bzero(a, sizeof(*a));
}

int GpuArray_share(GpuArray *a, GpuArray *b) {
  if (a->ops != b->ops) return 0;
  return a->ops->buffer_share(a->data, b->data, NULL);
}

int GpuArray_move(GpuArray *dst, GpuArray *src) {
  if (dst->ops != src->ops)
    return GA_INVALID_ERROR;
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst) || !GpuArray_ISONESEGMENT(src) || 
      GpuArray_ISFORTRAN(dst) != GpuArray_ISFORTRAN(src) || 
      GpuArray_ITEMSIZE(dst) != GpuArray_ITEMSIZE(src)) {
    return dst->ops->buffer_elemwise(src->data, dst->data, src->typecode,
				     dst->typecode, "=", src->nd,
				     src->dimensions, src->strides, dst->nd,
				     dst->dimensions, dst->strides);
  }
  return dst->ops->buffer_move(dst->data, src->data);
}

int GpuArray_write(GpuArray *dst, void *src, size_t src_sz) {
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst))
    return GA_UNSUPPORTED_ERROR;
  return dst->ops->buffer_write(dst->data, src, src_sz);
}

int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src) {
  if (!GpuArray_ISONESEGMENT(src))
    return GA_UNSUPPORTED_ERROR;
  return src->ops->buffer_read(dst, src->data, dst_sz);
}

int GpuArray_memset(GpuArray *a, int data) {
  if (!GpuArray_ISONESEGMENT(a))
    return GA_UNSUPPORTED_ERROR;
  return a->ops->buffer_memset(a->data, data);
}

const char *GpuArray_error(GpuArray *a, int err) {
  switch (err) {
  case GA_NO_ERROR:          return "No error";
  case GA_MEMORY_ERROR:      return "Out of memory";
  case GA_VALUE_ERROR:       return "Value out of range";
  case GA_IMPL_ERROR:        return a->ops->buffer_error();
  case GA_INVALID_ERROR:     return "Invalid value";
  case GA_UNSUPPORTED_ERROR: return "Unsupported operation";
  case GA_SYS_ERROR:         return strerror(errno);
  default: return "Unknown GA error";
  }
}

void GpuArray_fprintf(FILE *fd, const GpuArray *a) {
  int i;

  fprintf(fd, "GpuNdArray <%p, %p> nd=%d\n", a, a->data, a->nd);
  fprintf(fd, "\tITEMSIZE: %zd\n", GpuArray_ITEMSIZE(a));
  fprintf(fd, "\tTYPECODE: %d\n", a->typecode);
  fprintf(fd, "\tHOST_DIMS:      ");
  for (i = 0; i < a->nd; ++i)
    {
      fprintf(fd, "%zd\t", a->dimensions[i]);
    }
  fprintf(fd, "\n\tHOST_STRIDES: ");
  for (i = 0; i < a->nd; ++i)
    {
      fprintf(fd, "%zd\t", a->strides[i]);
    }
  fprintf(fd, "\nFLAGS:");
  int comma = 0;
#define PRINTFLAG(flag) if (a->flags & flag) { \
    if (comma) fputc(',', fd);                \
    fprintf(fd, " " #flag);                   \
    comma = 1;                                \
  }
  PRINTFLAG(GA_C_CONTIGUOUS);
  PRINTFLAG(GA_F_CONTIGUOUS);
  PRINTFLAG(GA_OWNDATA);
  PRINTFLAG(GA_ALIGNED);
  PRINTFLAG(GA_WRITEABLE);
#undef PRINTFLAG
  fputc('\n', fd);
}

int GpuArray_is_c_contiguous(const GpuArray *a) {
  size_t size = GpuArray_ITEMSIZE(a);
  int i;
  
  for (i = a->nd - 1; i >= 0; i--) {
    if (a->strides[i] != size) return 0;
    // We suppose that overflow will not happen since data has to fit in memory
    size *= a->dimensions[i];
  }
  return 1;
}

int GpuArray_is_f_contiguous(const GpuArray *a) {
  size_t size = GpuArray_ITEMSIZE(a);
  int i;

  for (i = 0; i < a->nd; i++) {
    if (a->strides[i] != size) return 0;
    // We suppose that overflow will not happen since data has to fit in memory
    size *= a->dimensions[i];
  }
  return 1;
}
