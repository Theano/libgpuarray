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
#include "gpuarray/array.h"
#include "gpuarray/error.h"
#include "gpuarray/util.h"

/*
 * Returns the boundaries of an array.
 *
 * This function works on virtual addresses where 0 is the start of
 * the gpu buffer and `offset` is the address of the first (0, ..., 0)
 * element of the array.  If you do not pass offset correctly, this
 * function will most likely overflow and return garbage results.
 *
 * On exit `start` holds the lowest (virtual) address ever touched by
 * the array and `end` holds the highest (virtual) address touched.
 * If you want the size of the memory region (to copy the data) you
 * need to add the size of one element to `end - start`.
 */
static void ga_boundaries(size_t *start, size_t *end, size_t offset,
                          unsigned int nd, size_t *dims, ssize_t *strs) {
  unsigned int i;
  *start = offset;
  *end = offset;

  for (i = 0; i < nd; i++) {
    if (dims[i] == 0) {
      *start = *end = offset;
      break;
    }

    if (strs[i] < 0)
      *start += (dims[i] - 1) * strs[i];
    else
      *end += (dims[i] - 1) * strs[i];
  }
}

/* Value below which a size_t multiplication will never overflow. */
#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

int GpuArray_empty(GpuArray *a, const gpuarray_buffer_ops *ops, void *ctx,
		   int typecode, unsigned int nd, const size_t *dims,
                   ga_order ord) {
  size_t size = gpuarray_get_elsize(typecode);
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

  size = gpuarray_get_elsize(typecode);
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

int GpuArray_zeros(GpuArray *a, const gpuarray_buffer_ops *ops, void *ctx,
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

int GpuArray_fromdata(GpuArray *a, const gpuarray_buffer_ops *ops,
                      gpudata *data, size_t offset, int typecode,
                      unsigned int nd, const size_t *dims,
                      const ssize_t *strides, int writeable) {
  if (gpuarray_get_type(typecode)->typecode != typecode)
    return GA_VALUE_ERROR;
  a->ops = ops;
  assert(data != NULL);
  a->data = data;
  ops->buffer_retain(a->data);
  a->nd = nd;
  a->offset = offset;
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  a->flags = (writeable ? GA_WRITEABLE : 0);
  if (a->dimensions == NULL || a->strides == NULL) {
    GpuArray_clear(a);
    return GA_MEMORY_ERROR;
  }
  memcpy(a->dimensions, dims, nd*sizeof(size_t));
  memcpy(a->strides, strides, nd*sizeof(ssize_t));

  if (GpuArray_is_c_contiguous(a)) a->flags |= GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a)) a->flags |= GA_F_CONTIGUOUS;
  if (GpuArray_is_aligned(a)) a->flags |= GA_ALIGNED;

  return GA_NO_ERROR;
}

int GpuArray_copy_from_host(GpuArray *a, const gpuarray_buffer_ops *ops,
                            void *ctx, void *buf, int typecode,
                            unsigned int nd, const size_t *dims,
                            const ssize_t *strides) {
  char *base = (char *)buf;
  size_t offset = 0;
  size_t size = gpuarray_get_elsize(typecode);
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

int GpuArray_index_inplace(GpuArray *a, const ssize_t *starts,
                           const ssize_t *stops, const ssize_t *steps) {
  unsigned int i, new_i;
  unsigned int new_nd = a->nd;
  size_t *newdims;
  ssize_t *newstrs;
  size_t new_offset = a->offset;

  if ((starts == NULL) || (stops == NULL) || (steps == NULL))
    return GA_VALUE_ERROR;

  for (i = 0; i < a->nd; i++) {
    if (steps[i] == 0) new_nd -= 1;
  }
  newdims = calloc(new_nd, sizeof(size_t));
  newstrs = calloc(new_nd, sizeof(ssize_t));
  if (newdims == NULL || newstrs == NULL) {
    free(newdims);
    free(newstrs);
    return GA_MEMORY_ERROR;
  }

  new_i = 0;
  for (i = 0; i < a->nd; i++) {
    if (starts[i] < -1 || (starts[i] > 0 &&
			   (size_t)starts[i] > a->dimensions[i])) {
      free(newdims);
      free(newstrs);
      return GA_VALUE_ERROR;
    }
    if (steps[i] == 0 &&
	(starts[i] == -1 || starts[i] >= a->dimensions[i])) {
      free(newdims);
      free(newstrs);
      return GA_VALUE_ERROR;
    }
    new_offset += starts[i] * a->strides[i];
    if (steps[i] != 0) {
      if ((stops[i] < -1 || (stops[i] > 0 &&
			      (size_t)stops[i] > a->dimensions[i])) ||
	  (stops[i]-starts[i])/steps[i] < 0) {
        free(newdims);
        free(newstrs);
	return GA_VALUE_ERROR;
      }
      newstrs[new_i] = steps[i] * a->strides[i];
      newdims[new_i] = (stops[i]-starts[i]+steps[i]-
                        (steps[i] < 0? -1 : 1))/steps[i];
      new_i++;
    }
  }
  a->nd = new_nd;
  a->offset = new_offset;
  free(a->dimensions);
  a->dimensions = newdims;
  free(a->strides);
  a->strides = newstrs;
  if (GpuArray_is_c_contiguous(a))
    a->flags |= GA_C_CONTIGUOUS;
  else
    a->flags &= ~GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a))
    a->flags |= GA_F_CONTIGUOUS;
  else
    a->flags &= ~GA_F_CONTIGUOUS;
  if (GpuArray_is_aligned(a))
    a->flags |= GA_ALIGNED;
  else
    a->flags &= ~GA_ALIGNED;

  return GA_NO_ERROR;
}

int GpuArray_index(GpuArray *r, const GpuArray *a, const ssize_t *starts,
                   const ssize_t *stops, const ssize_t *steps) {
  int err;
  err = GpuArray_view(r, a);
  if (err != GA_NO_ERROR) return err;
  err = GpuArray_index_inplace(r, starts, stops, steps);
  if (err != GA_NO_ERROR) GpuArray_clear(r);
  return err;
}

int GpuArray_setarray(GpuArray *a, const GpuArray *v) {
  unsigned int i;
  unsigned int off;
  int err = GA_NO_ERROR;
  size_t sz;
  ssize_t *strs;
  int simple_move = 1;


  if (a->nd < v->nd)
    return GA_VALUE_ERROR;

  if (a->ops != v->ops)
    return GA_INVALID_ERROR;
  if (!GpuArray_ISWRITEABLE(a))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISALIGNED(v) || !GpuArray_ISALIGNED(a))
    return GA_UNALIGNED_ERROR;

  off = a->nd - v->nd;

  for (i = 0; i < v->nd; i++) {
    if (v->dimensions[i] != a->dimensions[i+off]) {
      if (v->dimensions[i] != 1)
	return GA_VALUE_ERROR;
      else
	simple_move = 0;
    }
  }

  if (simple_move && GpuArray_ISONESEGMENT(a) && GpuArray_ISONESEGMENT(v) &&
      GpuArray_ISFORTRAN(a) == GpuArray_ISFORTRAN(v) &&
      a->typecode == v->typecode &&
      a->nd == v->nd) {
    sz = gpuarray_get_elsize(a->typecode);
    for (i = 0; i < a->nd; i++) sz *= a->dimensions[i];
    return a->ops->buffer_move(a->data, a->offset, v->data, v->offset, sz);
  }

  strs = calloc(a->nd, sizeof(ssize_t));
  if (strs == NULL)
    return GA_MEMORY_ERROR;

  for (i = off; i < a->nd; i++) {
    if (v->dimensions[i-off] == a->dimensions[i]) {
      strs[i] = v->strides[i-off];
    }
  }

  err = a->ops->buffer_extcopy(v->data, v->offset, a->data, a->offset,
			       v->typecode, a->typecode, a->nd, a->dimensions,
			       strs, a->nd, a->dimensions, a->strides);
  free(strs);
  return err;
}

int GpuArray_reshape(GpuArray *res, const GpuArray *a, unsigned int nd,
                     const size_t *newdims, ga_order ord, int nocopy) {
  int err;
  err = GpuArray_view(res, a);
  if (err != GA_NO_ERROR) return err;
  err = GpuArray_reshape_inplace(res, nd, newdims, ord);
  if (err == GA_COPY_ERROR && !nocopy) {
    GpuArray_clear(res);
    GpuArray_copy(res, a, ord);
    err = GpuArray_reshape_inplace(res, nd, newdims, ord);
  }
  if (err != GA_NO_ERROR) GpuArray_clear(res);
  return err;
}

int GpuArray_reshape_inplace(GpuArray *a, unsigned int nd,
                             const size_t *newdims, ga_order ord) {
  ssize_t *newstrides;
  size_t *tmpdims;
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
      newstrides[i] = gpuarray_get_elsize(a->typecode);
    }
  }

  /* We can reuse newstrides since it was allocated in this function.
     Can't do the same with newdims (which is a parameter). */
  tmpdims = calloc(nd, sizeof(size_t));
  if (tmpdims == NULL) {
    return GA_MEMORY_ERROR;
  }
  memcpy(tmpdims, newdims, nd*sizeof(size_t));
  a->nd = nd;
  free(a->dimensions);
  free(a->strides);
  a->dimensions = tmpdims;
  a->strides = newstrides;

  goto fix_flags;
 need_copy:
  free(newstrides);
  return GA_COPY_ERROR;

 do_final_copy:
  tmpdims = calloc(nd, sizeof(size_t));
  newstrides = calloc(nd, sizeof(ssize_t));
  if (tmpdims == NULL || newstrides == NULL) {
    free(tmpdims);
    free(newstrides);
    return GA_MEMORY_ERROR;
  }
  memcpy(tmpdims, newdims, nd*sizeof(size_t));
  if (nd > 0) {
    if (ord == GA_F_ORDER) {
      newstrides[0] = gpuarray_get_elsize(a->typecode);
      for (i = 1; i < nd; i++) {
        newstrides[i] = newstrides[i-1] * tmpdims[i-1];
      }
    } else {
      newstrides[nd-1] = gpuarray_get_elsize(a->typecode);
      for (i = nd-1; i > 0; i--) {
        newstrides[i-1] = newstrides[i] * tmpdims[i];
      }
    }
  }
  free(a->dimensions);
  free(a->strides);
  a->nd = nd;
  a->dimensions = tmpdims;
  a->strides = newstrides;

 fix_flags:
  if (GpuArray_is_c_contiguous(a))
    a->flags |= GA_C_CONTIGUOUS;
  else
    a->flags &= ~GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a))
    a->flags |= GA_F_CONTIGUOUS;
  else
    a->flags &= ~GA_F_CONTIGUOUS;
  if (GpuArray_is_aligned(a))
    a->flags |= GA_ALIGNED;
  else
    a->flags &= ~GA_ALIGNED;
  return GA_NO_ERROR;
}


int GpuArray_transpose(GpuArray *res, const GpuArray *a,
                       const unsigned int *new_axes) {
  int err;
  err = GpuArray_view(res, a);
  if (err != GA_NO_ERROR) return err;
  err = GpuArray_transpose_inplace(res, new_axes);
  if (err != GA_NO_ERROR) GpuArray_clear(res);
  return err;
}

int GpuArray_transpose_inplace(GpuArray *a, const unsigned int *new_axes) {
  size_t *newdims;
  ssize_t *newstrs;
  unsigned int i;
  unsigned int j;
  unsigned int k;

  newdims = calloc(a->nd, sizeof(size_t));
  newstrs = calloc(a->nd, sizeof(ssize_t));
  if (newdims == NULL || newstrs == NULL) {
    free(newdims);
    free(newstrs);
    return GA_MEMORY_ERROR;
  }

  for (i = 0; i < a->nd; i++) {
    if (new_axes == NULL) {
      j = a->nd - i - 1;
    } else {
      j = new_axes[i];
      // Repeated axes will lead to a broken output
      for (k = 0; k < i; k++)
        if (j == new_axes[k]) {
          free(newdims);
          free(newstrs);
          return GA_VALUE_ERROR;
        }
    }
    newdims[i] = a->dimensions[j];
    newstrs[i] = a->strides[j];
  }

  free(a->dimensions);
  free(a->strides);
  a->dimensions = newdims;
  a->strides = newstrs;

  a->flags &= ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
  if (GpuArray_is_c_contiguous(a))
    a->flags |= GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a))
    a->flags |= GA_F_CONTIGUOUS;

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
  if (a->ops != b->ops || a->data != b->data) return 0;
  /* XXX: redefine buffer_share to mean: is it possible to share?
          and use offset to make sure */
  return a->ops->buffer_share(a->data, b->data, NULL);
}

void *GpuArray_context(const GpuArray *a) {
  void *res = NULL;
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
  if (!GpuArray_ISALIGNED(src) || !GpuArray_ISALIGNED(dst))
    return GA_UNALIGNED_ERROR;
  if (!GpuArray_ISONESEGMENT(dst) || !GpuArray_ISONESEGMENT(src) ||
      GpuArray_ISFORTRAN(dst) != GpuArray_ISFORTRAN(src) ||
      dst->typecode != src->typecode ||
      dst->nd != src->nd) {
    return dst->ops->buffer_extcopy(src->data, src->offset, dst->data,
                                    dst->offset, src->typecode, dst->typecode,
                                    src->nd, src->dimensions, src->strides,
                                    dst->nd, dst->dimensions, dst->strides);
  }
  sz = gpuarray_get_elsize(dst->typecode);
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

int GpuArray_transfer(GpuArray *res, const GpuArray *a, void *new_ctx,
                      const gpuarray_buffer_ops *new_ops, int may_share) {
  size_t start, end;
  gpudata *tmp;
  int err;

  ga_boundaries(&start, &end, a->offset, a->nd, a->dimensions, a->strides);
  end += GpuArray_ITEMSIZE(a);

  tmp = gpuarray_buffer_transfer(a->data, start, end - start,
                                GpuArray_context(a), a->ops,
                                new_ctx, new_ops, may_share, &err);
  if (tmp == NULL)
    return err;

  return GpuArray_fromdata(res, new_ops, tmp, a->offset - start, a->typecode,
                           a->nd, a->dimensions, a->strides, 1);
}

int GpuArray_split(GpuArray **rs, const GpuArray *a, size_t n, size_t *p,
                   unsigned int axis) {
  size_t i;
  ssize_t *starts, *stops, *steps;
  int err;

  starts = calloc(a->nd, sizeof(ssize_t));
  stops = calloc(a->nd, sizeof(ssize_t));
  steps = calloc(a->nd, sizeof(ssize_t));

  if (starts == NULL || stops == NULL || steps == NULL) {
    free(starts);
    free(stops);
    free(steps);
    return GA_MEMORY_ERROR;
  }

  for (i = 0; i < a->nd; i++) {
    starts[i] = 0;
    stops[i] = a->dimensions[i];
    steps[i] = 1;
  }

  for (i = 0; i <= n; i++) {
    if (i > 0)
      starts[axis] = p[i-1];
    else
      starts[axis] = 0;
    if (i < n)
      stops[axis] = p[i];
    else
      stops[axis] = a->dimensions[axis];
    err = GpuArray_index(rs[i], a, starts, stops, steps);
    if (err != GA_NO_ERROR)
      break;
  }

  free(starts);
  free(stops);
  free(steps);

  if (err != GA_NO_ERROR) {
    size_t ii;
    for (ii = 0; ii < i; ii++)
      GpuArray_clear(rs[ii]);
  }
  return err;
}

int GpuArray_concatenate(GpuArray *r, const GpuArray **as, size_t n,
                         unsigned int axis, int restype) {
  size_t *dims;
  size_t i, j;
  size_t res_off;
  size_t sz;
  unsigned int p;
  int err = GA_NO_ERROR;

  dims = calloc(as[0]->nd, sizeof(size_t));
  if (dims == NULL)
    return GA_MEMORY_ERROR;

  for (p = 0; p < as[0]->nd; p++) {
    dims[p] = as[0]->dimensions[p];
  }

  if (!GpuArray_ISALIGNED(as[0])) {
    err = GA_UNALIGNED_ERROR;
    goto afterloop;
  }
  for (i = 1; i < n; i++) {
    if (!GpuArray_ISALIGNED(as[i])) {
      err = GA_UNALIGNED_ERROR;
      goto afterloop;
    }
    if (as[i]->nd != as[0]->nd) {
      err = GA_VALUE_ERROR;
      goto afterloop;
    }
    for (p = 0; p < as[0]->nd; p++) {
      if (p != axis && dims[p] != as[i]->dimensions[p]) {
        err = GA_VALUE_ERROR;
        goto afterloop;
      } else if (p == axis) {
        dims[p] += as[i]->dimensions[p];
      }
    }
  }

 afterloop:
  if (err != GA_NO_ERROR) {
    free(dims);
    return err;
  }

  err = GpuArray_empty(r, as[0]->ops, GpuArray_context(as[0]), restype,
                       as[0]->nd, dims, GA_C_ORDER);
  free(dims);
  if (err != GA_NO_ERROR) {
    return err;
  }

  res_off = 0;
  for (i = 0; i < n; i++) {
    sz = gpuarray_get_elsize(restype);
    for (j = 0; j < as[i]->nd; j++) sz *= as[i]->dimensions[j];

    if (!GpuArray_ISONESEGMENT(as[i]) || GpuArray_ISFORTRAN(as[i]) ||
        as[i]->typecode != r->typecode) {
      err = r->ops->buffer_extcopy(as[i]->data, as[i]->offset, r->data,
                                   res_off, as[i]->typecode, r->typecode,
                                   as[i]->nd, as[i]->dimensions,
                                   as[i]->strides, r->nd, r->dimensions,
                                   r->strides);
      if (err != GA_NO_ERROR)
        goto fail;
    } else {
      err = r->ops->buffer_move(r->data, res_off, as[i]->data, as[i]->offset,
                                sz);
      if (err != GA_NO_ERROR)
        goto fail;
    }
    res_off += sz;
  }

  return GA_NO_ERROR;
 fail:
  GpuArray_clear(r);
  return err;
}

const char *GpuArray_error(const GpuArray *a, int err) {
  void *ctx;
  int err2 = a->ops->property(NULL, a->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err2 != GA_NO_ERROR) {
    /* If CUDA refuses to work after any kind of error in kernels
       there is not much we can do about it. */
    return gpuarray_error_str(err);
  }
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

int GpuArray_is_aligned(const GpuArray *a) {
  size_t align = gpuarray_get_type(a->typecode)->align;
  unsigned int i;

  if (a->offset % align != 0)
    return 0;

  for (i = 0; i < a->nd; i++) {
    if (a->strides[i] % align != 0) return 0;
  }
  return 1;
}
