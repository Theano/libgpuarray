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

#include "private.h"
#include "compyte/array.h"
#include "compyte/error.h"
#include "compyte/util.h"

/* Value below which a size_t multiplication will never overflow. */
#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

int GpuArray_empty(GpuArray *a, const compyte_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, const size_t *dims,
                   ga_order ord) {
  size_t size = compyte_get_elsize(typecode);
  unsigned int i;
  int res = GA_NO_ERROR;

  if (ops == NULL)
    return GA_INVALID_ERROR;

  if (ord == GA_ANY_ORDER)
    ord = GA_C_ORDER;

  if (ord != GA_C_ORDER && ord != GA_F_ORDER)
    return GA_VALUE_ERROR;

  for (i = 0; i < nd; i++) {
    size_t d = dims[i];
    /* Check for overflow */
    if ((d >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
	d > 0 && SIZE_MAX / d < size)
      return GA_VALUE_ERROR;
    size *= d;
  }

  a->ops = ops;
  a->data = a->ops->buffer_alloc(ctx, size, NULL, 0, &res);
  if (a->data == NULL) return res;
  a->nd = nd;
  a->offset = 0;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* F/C distinction comes later */
  a->flags = GA_BEHAVED;
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
    for (i = nd; i > 0; i--) {
      a->strides[i-1] = size;
      size *= a->dimensions[i-1];
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

int GpuArray_zeros(GpuArray *a, const compyte_buffer_ops *ops, void *ctx,
                   int typecode, unsigned int nd, const size_t *dims,
                   ga_order ord) {
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

int GpuArray_fromdata(GpuArray *a, const compyte_buffer_ops *ops,
                      gpudata *data, size_t offset, int typecode,
                      unsigned int nd, const size_t *dims,
                      const ssize_t *strides, int writeable) {
  a->ops = ops;
  assert(data != NULL);
  a->data = data;
  ops->buffer_retain(a->data);
  a->nd = nd;
  a->offset = offset;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* XXX: We assume that the buffer is aligned */
  a->flags = (writeable ? GA_WRITEABLE : 0)|GA_ALIGNED;
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

int GpuArray_copy_from_host(GpuArray *a, const compyte_buffer_ops *ops,
                            void *ctx, void *buf, int typecode,
                            unsigned int nd, const size_t *dims,
                            const ssize_t *strides) {
  char *base = (char *)buf;
  size_t offset = 0;
  size_t size = compyte_get_elsize(typecode);
  gpudata *b;
  int err;
  unsigned int i;

  for (i = 0; i < nd; i++) {
    if (dims[i] == 0) {
      size = 0;
      base = (char *)buf;
      break;
    }

    if (strides[i] < 0)
      base += (dims[i]-1) * strides[i];
    else
      size += (dims[i]-1) * strides[i];
  }
  offset = (char *)buf - base;
  size += offset;

  b = ops->buffer_alloc(ctx, size, base, GA_BUFFER_INIT, &err);
  if (b == NULL) return err;

  err = GpuArray_fromdata(a, ops, b, offset, typecode, nd, dims, strides, 1);
  ops->buffer_release(b);
  return err;
}

int GpuArray_view(GpuArray *v, const GpuArray *a) {
  v->ops = a->ops;
  v->data = a->data;
  v->ops->buffer_retain(a->data);
  v->nd = a->nd;
  v->offset = a->offset;
  v->typecode = a->typecode;
  v->flags = a->flags;
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

int GpuArray_sync(GpuArray *a) {
  return a->ops->buffer_sync(a->data);
}

int GpuArray_index(GpuArray *r, const GpuArray *a, const ssize_t *starts,
                   const ssize_t *stops, const ssize_t *steps) {
  unsigned int i, r_i;
  unsigned int new_nd = a->nd;

  if ((starts == NULL) || (stops == NULL) || (steps == NULL))
    return GA_VALUE_ERROR;

  for (i = 0; i < a->nd; i++) {
    if (steps[i] == 0) new_nd -= 1;
  }
  r->ops = a->ops;
  r->data = a->data;
  r->ops->buffer_retain(r->data);
  r->typecode = a->typecode;
  r->flags = a->flags;
  r->nd = new_nd;
  r->offset = a->offset;
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

int GpuArray_setarray(GpuArray *a, const GpuArray *v) {
  GpuArray tmp;
  int i;
  int err = GA_NO_ERROR;
  void *t;

  if (a->nd < v->nd)
    return GA_VALUE_ERROR;

  for (i = 0; i < v->nd; i++) {
    if (v->dimensions[i] != 1 && v->dimensions[i] != a->dimensions[i])
      return GA_VALUE_ERROR;
  }

  err = GpuArray_view(&tmp, v);
  if (err != GA_NO_ERROR)
    return err;

  for (i = 0; i < tmp.nd; i++) {
    if (tmp.dimensions[i] == 1 && a->dimensions[i] != 1) {
      tmp.strides[i] = 0;
      tmp.dimensions[i] = a->dimensions[i];
      // Mark non-contiguous if we make broadcast dimensions
      tmp.flags &= ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
    }
  }

  if (a->nd > tmp.nd) {
    t = realloc(tmp.dimensions, a->nd*sizeof(size_t));
    if (t == NULL) {
      err = GA_NO_ERROR;
      goto exit;
    }
    tmp.dimensions = (size_t *)t;
    t = realloc(tmp.strides, a->nd*sizeof(size_t));
    if (t == NULL) {
      err = GA_NO_ERROR;
      goto exit;
    }
    tmp.strides = (ssize_t *)t;
    for (i = tmp.nd; i < a->nd; i++) {
      tmp.dimensions[i] = a->dimensions[i];
      tmp.strides[i] = 0;
    }
    tmp.nd = a->nd;
      // Mark non-contiguous if we make broadcast dimensions
    tmp.flags &= ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
  }

  err = GpuArray_move(a, &tmp);
  exit:
  GpuArray_clear(&tmp);
  return err;
}

int GpuArray_reshape(GpuArray *res, const GpuArray *a, unsigned int nd,
                     const size_t *newdims, ga_order ord, int nocopy) {
  ssize_t *newstrides;
  size_t np;
  size_t op;
  size_t newsize = 1;
  size_t oldsize = 1;
  unsigned int ni = 0;
  unsigned int oi = 0;
  unsigned int nj = 1;
  unsigned int oj = 1;
  unsigned int nk;
  unsigned int ok;
  unsigned int i;
  int err;

  if (ord == GA_ANY_ORDER && GpuArray_ISFORTRAN(a) && a->nd > 1)
    ord = GA_F_ORDER;

  for (i = 0; i < a->nd; i++) {
    oldsize *= a->dimensions[i];
  }

  for (i = 0; i < nd; i++) {
    size_t d = newdims[i];
    /* Check for overflow */
    if ((d >= MUL_NO_OVERFLOW || newsize >= MUL_NO_OVERFLOW) &&
	d > 0 && SIZE_MAX / d < newsize)
      return GA_INVALID_ERROR;
    newsize *= d;
  }

  if (newsize != oldsize) return GA_INVALID_ERROR;

  res->ops = a->ops;
  res->data = a->data;
  res->ops->buffer_retain(res->data);
  res->offset = a->offset;
  res->typecode = a->typecode;
  res->flags = a->flags;

  /* If the source and desired layouts are the same, then just copy
     strides and dimensions */
  if ((ord == GA_C_ORDER && GpuArray_CHKFLAGS(a, GA_C_CONTIGUOUS)) ||
      (ord == GA_F_ORDER && GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS))) {
    goto do_final_copy;
  }

  newstrides = calloc(nd, sizeof(ssize_t));
  if (newstrides == NULL)
    return GA_MEMORY_ERROR;

  while (ni < nd && oi < a->nd) {
    np = newdims[ni];
    op = a->dimensions[oi];

    while (np != op) {
      if (np < op) {
        np *= newdims[nj++];
      } else {
        op *= a->dimensions[oj++];
      }
    }

    for (ok = oi; ok < oj - 1; ok++) {
      if (ord == GA_F_ORDER) {
        if (a->strides[ok+1] != a->dimensions[ok]*a->strides[ok])
          goto need_copy;
      } else {
        if (a->strides[ok] != a->dimensions[ok+1]*a->strides[ok+1])
          goto need_copy;
      }
    }

    if (ord == GA_F_ORDER) {
      newstrides[ni] = a->strides[oi];
      for (nk = ni + 1; nk < nj; nk++) {
        newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
      }
    } else {
      newstrides[nj-1] = a->strides[oj-1];
      for (nk = nj-1; nk > ni; nk--) {
        newstrides[nk-1] = newstrides[nk]*newdims[nk];
      }
    }
    ni = nj++;
    oi = oj++;
  }

  /* Fixup trailing ones */
  if (ord == GA_F_ORDER) {
    for (i = nj-1; i < nd; i++) {
      newstrides[i] = newstrides[i-1] * newdims[i-1];
    }
  } else {
    for (i = nj-1; i < nd; i++) {
      newstrides[i] = compyte_get_elsize(res->typecode);
    }
  }

  res->nd = nd;
  /* We reuse newstrides since it was allocated in this function.
     Can't do the same with newdims (which is a parameter). */
  res->strides = newstrides;
  res->dimensions = calloc(nd, sizeof(size_t));
  if (res->dimensions == NULL) {
    GpuArray_clear(res);
    return GA_MEMORY_ERROR;
  }
  memcpy(res->dimensions, newdims, res->nd*sizeof(size_t));

  goto fix_flags;
 need_copy:
  free(newstrides);
  GpuArray_clear(res);
  if (nocopy)
    return GA_VALUE_ERROR; /* Might want a better error code */

  err = GpuArray_copy(res, a, ord);
  if (err != GA_NO_ERROR) return err;
  free(res->dimensions);
  free(res->strides);

 do_final_copy:
  res->nd = nd;
  res->dimensions = calloc(res->nd, sizeof(size_t));
  res->strides = calloc(res->nd, sizeof(ssize_t));
  if (res->dimensions == NULL || res->strides == NULL) {
    GpuArray_clear(res);
    return GA_MEMORY_ERROR;
  }
  memcpy(res->dimensions, newdims, res->nd*sizeof(size_t));
  if (res->nd > 0) {
    if (ord == GA_F_ORDER) {
      res->strides[0] = compyte_get_elsize(res->typecode);
      for (i = 1; i < res->nd; i++) {
        res->strides[i] = res->strides[i-1] * res->dimensions[i-1];
      }
    } else {
      res->strides[res->nd-1] = compyte_get_elsize(res->typecode);
      for (i = res->nd-1; i > 0; i--) {
        res->strides[i-1] = res->strides[i] * res->dimensions[i];
      }
    }
  }

 fix_flags:
  if (GpuArray_is_c_contiguous(res))
    res->flags |= GA_C_CONTIGUOUS;
  else
    res->flags &= ~GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(res))
    res->flags |= GA_F_CONTIGUOUS;
  else
    res->flags &= ~GA_F_CONTIGUOUS;
  return GA_NO_ERROR;
}

int GpuArray_transpose(GpuArray *res, const GpuArray *a,
                       const unsigned int *new_axes) {
  unsigned int i;
  unsigned int j;
  unsigned int k;

  res->ops = a->ops;
  res->data = a->data;
  res->ops->buffer_retain(a->data);
  res->offset = a->offset;
  res->nd = a->nd;
  res->flags = a->flags & ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
  res->typecode = a->typecode;
  res->dimensions = calloc(res->nd, sizeof(size_t));
  res->strides = calloc(res->nd, sizeof(ssize_t));
  if (res->dimensions == NULL || res->strides == NULL) {
    GpuArray_clear(res);
    return GA_MEMORY_ERROR;
  }

  for (i = 0; i < res->nd; i++) {
    if (new_axes == NULL) {
      j = res->nd - i - 1;
    } else {
      j = new_axes[i];
      // Repeated axes will lead to a broken output
      for (k = 0; k < i; k++)
        if (j == new_axes[k]) {
          GpuArray_clear(res);
          return GA_VALUE_ERROR;
        }
    }
    res->dimensions[i] = a->dimensions[j];
    res->strides[i] = a->strides[j];
  }
  if (new_axes == NULL) {
    if (a->flags & GA_C_CONTIGUOUS) res->flags |= GA_F_CONTIGUOUS;
    if (a->flags & GA_F_CONTIGUOUS) res->flags |= GA_C_CONTIGUOUS;
  } else {
  if (GpuArray_is_c_contiguous(res))
    res->flags |= GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(res))
    res->flags |= GA_F_CONTIGUOUS;
  }

  return GA_NO_ERROR;
}

void GpuArray_clear(GpuArray *a) {
  if (a->data)
    a->ops->buffer_release(a->data);
  free(a->dimensions);
  free(a->strides);
  memset(a, 0, sizeof(*a));
}

int GpuArray_share(const GpuArray *a, const GpuArray *b) {
  if (a->ops != b->ops) return 0;
  /* XXX: redefine buffer_share to mean: is it possible to share?
          and use offset to make sure */
  return a->ops->buffer_share(a->data, b->data, NULL);
}

void *GpuArray_context(const GpuArray *a) {
  void *res;
  (void)a->ops->property(NULL, a->data, NULL, GA_BUFFER_PROP_CTX, &res);
  return res;
}

int GpuArray_move(GpuArray *dst, const GpuArray *src) {
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

int GpuArray_write(GpuArray *dst, const void *src, size_t src_sz) {
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst))
    return GA_UNSUPPORTED_ERROR;
  return dst->ops->buffer_write(dst->data, dst->offset, src, src_sz);
}

int GpuArray_read(void *dst, size_t dst_sz, const GpuArray *src) {
  if (!GpuArray_ISONESEGMENT(src))
    return GA_UNSUPPORTED_ERROR;
  return src->ops->buffer_read(dst, src->data, src->offset, dst_sz);
}

int GpuArray_memset(GpuArray *a, int data) {
  if (!GpuArray_ISONESEGMENT(a))
    return GA_UNSUPPORTED_ERROR;
  return a->ops->buffer_memset(a->data, a->offset, data);
}

int GpuArray_copy(GpuArray *res, const GpuArray *a, ga_order order) {
  int err;
  err = GpuArray_empty(res, a->ops, GpuArray_context(a), a->typecode,
                       a->nd, a->dimensions, order);
  if (err != GA_NO_ERROR) return err;
  err = GpuArray_move(res, a);
  if (err != GA_NO_ERROR)
    GpuArray_clear(res);
  return err;
}

const char *GpuArray_error(const GpuArray *a, int err) {
  void *ctx;
  a->ops->property(NULL, a->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  return Gpu_error(a->ops, ctx, err);
}

void GpuArray_fprintf(FILE *fd, const GpuArray *a) {
  unsigned int i;
  int comma = 0;

  fprintf(fd, "GpuNdArray <%p, %p, %p> nd=%d\n", a, a->data, a->ops, a->nd);
  fprintf(fd, "\tdims: %p, str: %p\n", a->dimensions, a->strides);
  fprintf(fd, "\tITEMSIZE: %zd\n", GpuArray_ITEMSIZE(a));
  fprintf(fd, "\tTYPECODE: %d\n", a->typecode);
  fprintf(fd, "\tOFFSET: %" SPREFIX "u\n", a->offset);
  fprintf(fd, "\tHOST_DIMS:      ");
  for (i = 0; i < a->nd; ++i) {
      fprintf(fd, "%zu\t", a->dimensions[i]);
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
