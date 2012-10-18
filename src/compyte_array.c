#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#ifndef _MSC_VER
#include <stdint.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "compyte/array.h"
#include "compyte/error.h"
#include "compyte/util.h"

#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

int GpuArray_empty(GpuArray *a, compyte_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, size_t *dims, ga_order ord) {
  size_t size = compyte_get_elsize(typecode);
  int i;
  int res = GA_NO_ERROR;

  if (ops == NULL)
    return GA_INVALID_ERROR;

  if (ord == GA_ANY_ORDER)
    ord = GA_C_ORDER;

  if (ord != GA_C_ORDER && ord != GA_F_ORDER)
    return GA_VALUE_ERROR;

  for (i = 0; (unsigned)i < nd; i++) {
    size_t d = dims[i];
    if ((d >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
	d > 0 && SIZE_MAX / d < size)
      return GA_VALUE_ERROR;
    size *= d;
  }

  a->ops = ops;
  a->data = a->ops->buffer_alloc(ctx, size, &res);
  if (res != GA_NO_ERROR) return res;
  a->nd = nd;
  a->offset = 0;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* F/C distinction comes later */
  a->flags = GA_OWNDATA|GA_BEHAVED;
  if (a->dimensions == NULL || a->strides == NULL) {
    GpuArray_clear(a);
    return GA_MEMORY_ERROR;
  }
  /* Mult will not overflow since calloc succeded */
  memcpy(a->dimensions, dims, sizeof(size_t)*nd);

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
    for (i = 0; (unsigned)i < nd; i++) {
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
  err = a->ops->buffer_memset(a->data, a->offset, 0);
  if (err != GA_NO_ERROR) {
    GpuArray_clear(a);
  }
  return err;
}

int GpuArray_fromdata(GpuArray *a, compyte_buffer_ops *ops, gpudata *data,
                      size_t offset, int typecode, unsigned int nd, size_t *dims,
                      ssize_t *strides, int writeable) {
  a->ops = ops;
  assert(data != NULL);
  a->data = data;
  a->nd = nd;
  a->offset = offset;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* XXX: We assume that the buffer is aligned */
  a->flags = GA_OWNDATA|(writeable ? GA_WRITEABLE : 0)|GA_ALIGNED;
  if (a->dimensions == NULL || a->strides == NULL) {
    GpuArray_clear(a);
    return GA_MEMORY_ERROR;
  }
  memcpy(a->dimensions, dims, nd*sizeof(size_t));
  memcpy(a->strides, strides, nd*sizeof(ssize_t));

  if (GpuArray_is_c_contiguous(a)) a->flags |= GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a)) a->flags |= GA_F_CONTIGUOUS;

  return GA_NO_ERROR;
}

int GpuArray_view(GpuArray *v, GpuArray *a) {
  v->ops = a->ops;
  v->data = a->data;
  v->nd = a->nd;
  v->offset = a->offset;
  v->typecode = a->typecode;
  v->flags = a->flags & ~GA_OWNDATA;
  v->dimensions = calloc(v->nd, sizeof(size_t));
  v->strides = calloc(v->nd, sizeof(ssize_t));
  if (v->dimensions == NULL || v->strides == NULL) {
    GpuArray_clear(v);
    return GA_MEMORY_ERROR;
  }
  memcpy(v->dimensions, a->dimensions, v->nd*sizeof(size_t));
  memcpy(v->strides, a->strides, v->nd*sizeof(ssize_t));
  return GA_NO_ERROR;
}

int GpuArray_index(GpuArray *r, GpuArray *a, ssize_t *starts, ssize_t *stops,
		   ssize_t *steps) {
  int err;
  unsigned int i, r_i;
  unsigned int new_nd = a->nd;

  if ((starts == NULL) || (stops == NULL) || (steps == NULL))
    return GA_VALUE_ERROR;

  for (i = 0; i < a->nd; i++) {
    if (steps[i] == 0) new_nd -= 1;
  }
  r->ops = a->ops;
  r->data = a->data;
  r->typecode = a->typecode;
  r->flags = a->flags & ~GA_OWNDATA;
  r->nd = new_nd;
  r->offset = a->offset;
  if (r->data == NULL) {
    GpuArray_clear(r);
    return err;
  }
  r->dimensions = calloc(r->nd, sizeof(size_t));
  r->strides = calloc(r->nd, sizeof(ssize_t));
  if (r->dimensions == NULL || r->strides == NULL) {
    GpuArray_clear(r);
    return GA_MEMORY_ERROR;
  }

  r_i = 0;
  for (i = 0; i < a->nd; i++) {
    if (starts[i] > 0 && (size_t)starts[i] >= a->dimensions[i]) {
      GpuArray_clear(r);
      return GA_VALUE_ERROR;
    }
    r->offset += starts[i] * a->strides[i];
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
  memset(a, 0, sizeof(*a));
}

int GpuArray_share(GpuArray *a, GpuArray *b) {
  if (a->ops != b->ops) return 0;
  /* XXX: redefine buffer_share to mean: is it possible to share?
          and use offset to make sure */
  return a->ops->buffer_share(a->data, b->data, NULL);
}

int GpuArray_move(GpuArray *dst, GpuArray *src) {
  size_t sz;
  unsigned int i;
  if (dst->ops != src->ops)
    return GA_INVALID_ERROR;
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst) || !GpuArray_ISONESEGMENT(src) ||
      GpuArray_ISFORTRAN(dst) != GpuArray_ISFORTRAN(src) ||
      dst->typecode != src->typecode ||
      dst->nd != src->nd) {
    return dst->ops->buffer_extcopy(src->data, src->offset, dst->data,
                                    dst->offset, src->typecode, dst->typecode,
                                    src->nd, src->dimensions, src->strides,
                                    dst->nd, dst->dimensions, dst->strides);
  }
  sz = compyte_get_elsize(dst->typecode);
  for (i = 0; i < dst->nd; i++) sz *= dst->dimensions[i];
  return dst->ops->buffer_move(dst->data, dst->offset, src->data, src->offset,
                               sz);
}

int GpuArray_write(GpuArray *dst, void *src, size_t src_sz) {
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst))
    return GA_UNSUPPORTED_ERROR;
  return dst->ops->buffer_write(dst->data, dst->offset, src, src_sz);
}

int GpuArray_read(void *dst, size_t dst_sz, GpuArray *src) {
  if (!GpuArray_ISONESEGMENT(src))
    return GA_UNSUPPORTED_ERROR;
  return src->ops->buffer_read(dst, src->data, src->offset, dst_sz);
}

int GpuArray_memset(GpuArray *a, int data) {
  if (!GpuArray_ISONESEGMENT(a))
    return GA_UNSUPPORTED_ERROR;
  return a->ops->buffer_memset(a->data, a->offset, data);
}

const char *GpuArray_error(GpuArray *a, int err) {
  return Gpu_error(a->ops, a->data, err);
}

void GpuArray_fprintf(FILE *fd, const GpuArray *a) {
  unsigned int i;
  int comma = 0;

  fprintf(fd, "GpuNdArray <%p, %p> nd=%d\n", a, a->data, a->nd);
  fprintf(fd, "\tITEMSIZE: %zd\n", GpuArray_ITEMSIZE(a));
  fprintf(fd, "\tTYPECODE: %d\n", a->typecode);
  fprintf(fd, "\tOFFSET: %" SPREFIX "u\n", a->offset);
  fprintf(fd, "\tHOST_DIMS:      ");
  for (i = 0; i < a->nd; ++i) {
      fprintf(fd, "%zd\t", a->dimensions[i]);
  }
  fprintf(fd, "\n\tHOST_STRIDES: ");
  for (i = 0; i < a->nd; ++i) {
      fprintf(fd, "%zd\t", a->strides[i]);
  }
  fprintf(fd, "\nFLAGS:");
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
  unsigned int i;

  for (i = 0; i < a->nd; i++) {
    if (a->strides[i] != size) return 0;
    // We suppose that overflow will not happen since data has to fit in memory
    size *= a->dimensions[i];
  }
  return 1;
}
