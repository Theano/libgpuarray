#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#if _MSC_VER < 1600
#include <stdint.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "private.h"
#include "gpuarray/array.h"
#include "gpuarray/error.h"
#include "gpuarray/kernel.h"
#include "gpuarray/util.h"

#include "util/strb.h"
#include "util/xxhash.h"

struct extcopy_args {
  int itype;
  int otype;
};

static int extcopy_eq(cache_key_t _k1, cache_key_t _k2) {
  struct extcopy_args *k1 = _k1;
  struct extcopy_args *k2 = _k2;
  return k1->itype == k2->itype && k1->otype == k2->otype;
}

static void extcopy_free(cache_key_t k) {
  free(k);
}

static uint32_t extcopy_hash(cache_key_t k) {
  return XXH32(k, sizeof(struct extcopy_args), 42);
}

static int ga_extcopy(GpuArray *dst, const GpuArray *src) {
  struct extcopy_args a, *aa;
  gpucontext *ctx = gpudata_context(dst->data);
  GpuElemwise *k = NULL;
  void *args[2];

  if (ctx != gpudata_context(src->data))
    return GA_INVALID_ERROR;

  a.itype = src->typecode;
  a.otype = dst->typecode;

  if (ctx->extcopy_cache != NULL)
    k = cache_get(ctx->extcopy_cache, &a);
  if (k == NULL) {
    gpuelemwise_arg gargs[2];
    gargs[0].name = "src";
    gargs[0].typecode = src->typecode;
    gargs[0].flags = GE_READ;
    gargs[1].name = "dst";
    gargs[1].typecode = dst->typecode;
    gargs[1].flags = GE_WRITE;
    k = GpuElemwise_new(ctx, "", "dst = src", 2, gargs, 0, 0);
    if (k == NULL)
      return GA_MISC_ERROR;
    aa = memdup(&a, sizeof(a));
    if (aa == NULL)
      return GA_MEMORY_ERROR;
    if (ctx->extcopy_cache == NULL)
      ctx->extcopy_cache = cache_twoq(4, 8, 8, 2, extcopy_eq, extcopy_hash,
                                      extcopy_free,
                                      (cache_freev_fn)GpuElemwise_free);
    if (ctx->extcopy_cache == NULL)
      return GA_MISC_ERROR;
    if (cache_add(ctx->extcopy_cache, aa, k) != 0)
      return GA_MISC_ERROR;
  }
  args[0] = (void *)src;
  args[1] = (void *)dst;
  return GpuElemwise_call(k, args, GE_BROADCAST);
}

/* Value below which a size_t multiplication will never overflow. */
#define MUL_NO_OVERFLOW (1UL << (sizeof(size_t) * 4))

int GpuArray_empty(GpuArray *a, gpucontext *ctx,
		   int typecode, unsigned int nd, const size_t *dims,
                   ga_order ord) {
  size_t size = gpuarray_get_elsize(typecode);
  unsigned int i;
  int res = GA_NO_ERROR;

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

  a->data = gpudata_alloc(ctx, size, NULL, 0, &res);
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

int GpuArray_zeros(GpuArray *a, gpucontext *ctx,
                   int typecode, unsigned int nd, const size_t *dims,
                   ga_order ord) {
  int err;
  err = GpuArray_empty(a, ctx, typecode, nd, dims, ord);
  if (err != GA_NO_ERROR)
    return err;
  err = gpudata_memset(a->data, a->offset, 0);
  if (err != GA_NO_ERROR) {
    GpuArray_clear(a);
  }
  return err;
}

int GpuArray_fromdata(GpuArray *a, gpudata *data, size_t offset, int typecode,
                      unsigned int nd, const size_t *dims,
                      const ssize_t *strides, int writeable) {
  if (gpuarray_get_type(typecode)->typecode != typecode)
    return GA_VALUE_ERROR;
  assert(data != NULL);
  a->data = data;
  gpudata_retain(a->data);
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

int GpuArray_copy_from_host(GpuArray *a, gpucontext *ctx, void *buf,
                            int typecode, unsigned int nd, const size_t *dims,
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

  b = gpudata_alloc(ctx, size, base, GA_BUFFER_INIT, &err);
  if (b == NULL) return err;

  err = GpuArray_fromdata(a, b, offset, typecode, nd, dims, strides, 1);
  gpudata_release(b);
  return err;
}

int GpuArray_view(GpuArray *v, const GpuArray *a) {
  v->data = a->data;
  gpudata_retain(a->data);
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
  return gpudata_sync(a->data);
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

static int gen_take1_kernel(GpuKernel *k, gpucontext *ctx, char **err_str,
                            GpuArray *a, const GpuArray *v,
                            const GpuArray *ind, int addr32) {
  strb sb = STRB_STATIC_INIT;
  int *atypes;
  size_t nargs, apos;
  char *sz, *ssz;
  unsigned int i, i2;
  int flags = GA_USE_CLUDA;
  int res;

  nargs = 7 + 2 * v->nd;

  atypes = calloc(nargs, sizeof(int));
  if (atypes == NULL)
    return GA_MEMORY_ERROR;

  if (addr32) {
    sz = "ga_uint";
    ssz = "ga_int";
  } else {
    sz = "ga_size";
    ssz = "ga_ssize";
  }

  apos = 0;
  strb_appendf(&sb, "KERNEL void take1(GLOBAL_MEM %s *r, "
               "GLOBAL_MEM const %s *v, ga_size off,",
               gpuarray_get_type(a->typecode)->cluda_name,
               gpuarray_get_type(v->typecode)->cluda_name);
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_SIZE;
  for (i = 0; i < v->nd; i++) {
    strb_appendf(&sb, " ga_ssize s%u, ga_size d%u,", i, i);
    atypes[apos++] = GA_SSIZE;
    atypes[apos++] = GA_SIZE;
  }
  strb_appends(&sb, " GLOBAL_MEM const ga_ssize *ind, ga_size n0, ga_size n1,"
               " GLOBAL_MEM int* err) {\n");
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_BUFFER;
  assert(apos == nargs);
  strb_appendf(&sb, "  const %s idx0 = LDIM_0 * GID_0 + LID_0;\n"
               "  const %s numThreads0 = LDIM_0 * GDIM_0;\n"
               "  const %s idx1 = LDIM_1 * GID_1 + LID_1;\n"
               "  const %s numThreads1 = LDIM_1 * GDIM_1;\n"
               "  %s i0, i1;\n", sz, sz, sz, sz, sz);
  strb_appendf(&sb, "  for (i0 = idx0; i0 < n0; i0 += numThreads0) {\n"
               "    %s ii0 = ind[i0];\n"
               "    %s pos0 = off;\n"
               "    if (ii0 < 0) ii0 += d0;\n"
               "    if ((ii0 < 0) || (ii0 >= d0)) {\n"
               "      *err = -1;\n"
               "      continue;\n"
               "    }\n"
               "    pos0 += ii0 * (%s)s0;\n"
               "    for (i1 = idx1; i1 < n1; i1 += numThreads1) {\n"
               "      %s p = pos0;\n", ssz, sz, sz, sz);
  if (v->nd > 1) {
    strb_appendf(&sb, "      %s pos, ii = i1;\n", sz);
    for (i2 = v->nd; i2 > 1; i2--) {
      i = i2 - 1;
      if (i > 1)
        strb_appendf(&sb, "      pos = ii %% (%s)d%u;\n"
                     "      ii /= (%s)d%u;\n", sz, i, sz, i);
      else
        strb_appends(&sb, "      pos = ii;\n");
      strb_appendf(&sb, "      p += pos * (%s)s%u;\n", ssz, i);
    }
  }
  strb_appendf(&sb, "      r[i0*((%s)n1) + i1] = *((GLOBAL_MEM %s *)(((GLOBAL_MEM char *)v) + p));\n",
               sz, gpuarray_get_type(v->typecode)->cluda_name);
  strb_appends(&sb, "    }\n"
               "  }\n"
               "}\n");
  if (strb_error(&sb)) {
    res = GA_MEMORY_ERROR;
    goto bail;
  }
  flags |= gpuarray_type_flags(a->typecode, v->typecode, GA_BYTE, -1);
  res = GpuKernel_init(k, ctx, 1, (const char **)&sb.s, &sb.l, "take1",
                       nargs, atypes, flags, err_str);
bail:
  free(atypes);
  strb_clear(&sb);
  return res;
}

int GpuArray_take1(GpuArray *a, const GpuArray *v, const GpuArray *i,
                   int check_error) {
  size_t n[2], ls[2] = {0, 0}, gs[2] = {0, 0};
  size_t pl;
  gpudata *errbuf;
#if DEBUG
  char *errstr = NULL;
#endif
  size_t argp;
  GpuKernel k;
  unsigned int j;
  int err, kerr = 0;
  int addr32 = 0;

  if (!GpuArray_ISWRITEABLE(a))
    return GA_INVALID_ERROR;

  if (!GpuArray_ISALIGNED(a) || !GpuArray_ISALIGNED(v) ||
      !GpuArray_ISALIGNED(i))
    return GA_UNALIGNED_ERROR;

  /* a and i have to be C contiguous */
  if (!GpuArray_IS_C_CONTIGUOUS(a) || !GpuArray_IS_C_CONTIGUOUS(i))
    return GA_INVALID_ERROR;

  /* Check that the dimensions match namely a[0] == i[0] and a[>0] == v[>0] */
  if (v->nd == 0 || a->nd == 0 || i->nd != 1 || a->nd != v->nd ||
      a->dimensions[0] != i->dimensions[0])
    return GA_INVALID_ERROR;

  n[0] = i->dimensions[0];
  n[1] = 1;

  for (j = 1; j < v->nd; j++) {
    if (a->dimensions[j] != v->dimensions[j])
      return GA_INVALID_ERROR;
    n[1] *= v->dimensions[j];
  }

  if (n[0] * n[1] < SADDR32_MAX) {
    addr32 = 1;
  }

  err = gpudata_property(v->data, GA_CTX_PROP_ERRBUF, &errbuf);
  if (err != GA_NO_ERROR)
    return err;

  err = gen_take1_kernel(&k, GpuArray_context(a),
#if DEBUG
                         &errstr,
#else
                         NULL,
#endif
                         a, v, i, addr32);
#if DEBUG
  if (errstr != NULL) {
    fprintf(stderr, "%s\n", errstr);
    free(errstr);
  }
#endif
  if (err != GA_NO_ERROR)
    return err;

  err = GpuKernel_sched(&k, n[0]*n[1], &ls[1], &gs[1]);
  if (err != GA_NO_ERROR)
    goto out;

  /* This may not be the best scheduling, but it's good enough */
  err = gpukernel_property(k.k, GA_KERNEL_PROP_PREFLSIZE, &pl);
  ls[0] = ls[1] / pl;
  ls[1] = pl;
  if (n[1] > n[0]) {
    pl = ls[0];
    ls[0] = ls[1];
    ls[1] = pl;
  }
  gs[0] = 1;

  argp = 0;
  GpuKernel_setarg(&k, argp++, a->data);
  GpuKernel_setarg(&k, argp++, v->data);
  GpuKernel_setarg(&k, argp++, (void *)&v->offset);
  for (j = 0; j < v->nd; j++) {
    GpuKernel_setarg(&k, argp++, &v->strides[j]);
    GpuKernel_setarg(&k, argp++, &v->dimensions[j]);
  }
  GpuKernel_setarg(&k, argp++, i->data);
  GpuKernel_setarg(&k, argp++, &n[0]);
  GpuKernel_setarg(&k, argp++, &n[1]);
  GpuKernel_setarg(&k, argp++, errbuf);

  err = GpuKernel_call(&k, 2, ls, gs, 0, NULL);
  if (check_error && err == GA_NO_ERROR) {
    err = gpudata_read(&kerr, errbuf, 0, sizeof(int));
    if (err == GA_NO_ERROR && kerr != 0) {
      err = GA_VALUE_ERROR;
      kerr = 0;
      /* We suppose this will not fail */
      gpudata_write(errbuf, 0, &kerr, sizeof(int));
    }
  }

out:
  GpuKernel_clear(&k);
  return err;
}

int GpuArray_setarray(GpuArray *a, const GpuArray *v) {
  GpuArray tv;
  size_t sz;
  ssize_t *strs;
  unsigned int i, off;
  int err = GA_NO_ERROR;
  int simple_move = 1;

  if (a->nd < v->nd)
    return GA_VALUE_ERROR;

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
    return gpudata_move(a->data, a->offset, v->data, v->offset, sz);
  }

  strs = calloc(a->nd, sizeof(ssize_t));
  if (strs == NULL)
    return GA_MEMORY_ERROR;

  for (i = off; i < a->nd; i++) {
    if (v->dimensions[i-off] == a->dimensions[i]) {
      strs[i] = v->strides[i-off];
    }
  }

  memcpy(&tv, v, sizeof(GpuArray));
  tv.nd = a->nd;
  tv.dimensions = a->dimensions;
  tv.strides = strs;
  /* This could be optiomized by setting the right flags */
  tv.flags &= ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
  err = ga_extcopy(a, &tv);
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
    gpudata_release(a->data);
  free(a->dimensions);
  free(a->strides);
  memset(a, 0, sizeof(*a));
}

int GpuArray_share(const GpuArray *a, const GpuArray *b) {
  if (a->data != b->data) return 0;
  /* XXX: redefine buffer_share to mean: is it possible to share?
          and use offset to make sure */
  return gpudata_share(a->data, b->data, NULL);
}

gpucontext *GpuArray_context(const GpuArray *a) {
  return gpudata_context(a->data);
}

int GpuArray_move(GpuArray *dst, const GpuArray *src) {
  size_t sz;
  unsigned int i;
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISALIGNED(src) || !GpuArray_ISALIGNED(dst))
    return GA_UNALIGNED_ERROR;
  if (src->nd != dst->nd)
    return GA_VALUE_ERROR;
  for (i = 0; i < src->nd; i++) {
    if (src->dimensions[i] != dst->dimensions[i])
      return GA_VALUE_ERROR;
  }
  if (!GpuArray_ISONESEGMENT(dst) || !GpuArray_ISONESEGMENT(src) ||
      GpuArray_ISFORTRAN(dst) != GpuArray_ISFORTRAN(src) ||
      dst->typecode != src->typecode) {
    return ga_extcopy(dst, src);
  }
  sz = gpuarray_get_elsize(dst->typecode);
  for (i = 0; i < dst->nd; i++) sz *= dst->dimensions[i];
  return gpudata_move(dst->data, dst->offset, src->data, src->offset, sz);
}

int GpuArray_write(GpuArray *dst, const void *src, size_t src_sz) {
  if (!GpuArray_ISWRITEABLE(dst))
    return GA_VALUE_ERROR;
  if (!GpuArray_ISONESEGMENT(dst))
    return GA_UNSUPPORTED_ERROR;
  return gpudata_write(dst->data, dst->offset, src, src_sz);
}

int GpuArray_read(void *dst, size_t dst_sz, const GpuArray *src) {
  if (!GpuArray_ISONESEGMENT(src))
    return GA_UNSUPPORTED_ERROR;
  return gpudata_read(dst, src->data, src->offset, dst_sz);
}

int GpuArray_memset(GpuArray *a, int data) {
  if (!GpuArray_ISONESEGMENT(a))
    return GA_UNSUPPORTED_ERROR;
  return gpudata_memset(a->data, a->offset, data);
}

int GpuArray_copy(GpuArray *res, const GpuArray *a, ga_order order) {
  int err;
  err = GpuArray_empty(res, GpuArray_context(a), a->typecode,
                       a->nd, a->dimensions, order);
  if (err != GA_NO_ERROR) return err;
  err = GpuArray_move(res, a);
  if (err != GA_NO_ERROR)
    GpuArray_clear(res);
  return err;
}

int GpuArray_transfer(GpuArray *res, const GpuArray *a) {
  size_t sz;
  unsigned int i;

  if (!GpuArray_ISONESEGMENT(res))
    return GA_UNSUPPORTED_ERROR;
  if (!GpuArray_ISONESEGMENT(a))
    return GA_UNSUPPORTED_ERROR;

  if (res->typecode != a->typecode)
    return GA_UNSUPPORTED_ERROR;

  sz = gpuarray_get_elsize(a->typecode);
  for (i = 0; i < a->nd; i++) sz *= a->dimensions[i];

 return gpudata_transfer(res->data, res->offset, a->data, a->offset, sz);
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
  size_t *dims, *res_dims;
  size_t i, res_off;
  unsigned int p;
  int res_flags;
  int err = GA_NO_ERROR;

  if (axis >= as[0]->nd)
    return GA_VALUE_ERROR;

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

  err = GpuArray_empty(r, GpuArray_context(as[0]), restype,
                       as[0]->nd, dims, GA_ANY_ORDER);
  free(dims);
  if (err != GA_NO_ERROR) {
    return err;
  }

  res_off = r->offset;
  res_dims = r->dimensions;
  res_flags = r->flags;
  /* This could be optimized by setting the right flags */
  r->flags &= ~(GA_C_CONTIGUOUS|GA_F_CONTIGUOUS);
  for (i = 0; i < n; i++) {
    r->dimensions = as[i]->dimensions;
    err = ga_extcopy(r, as[i]);
    if (err != GA_NO_ERROR) {
      r->dimensions = res_dims;
      goto fail;
    }
    r->offset += r->strides[axis] * as[i]->dimensions[axis];
  }
  r->offset = res_off;
  r->dimensions = res_dims;
  r->flags = res_flags;

  return GA_NO_ERROR;
 fail:
  GpuArray_clear(r);
  return err;
}

const char *GpuArray_error(const GpuArray *a, int err) {
  return gpucontext_error(gpudata_context(a->data), err);
}

void GpuArray_fprintf(FILE *fd, const GpuArray *a) {
  unsigned int i;
  int comma = 0;

  fprintf(fd, "GpuNdArray <%p, data: %p (%p)> nd=%d\n",
          a, a->data, *((void **)a->data), a->nd);
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

int GpuArray_fdump(FILE *fd, const GpuArray *a) {
  char *buf, *p;
  size_t s = GpuArray_ITEMSIZE(a);
  size_t k;
  unsigned int i;
  int err;

  for (i = 0; i < a->nd; i++)
    s *= a->dimensions[i];

  buf = malloc(s);
  if (buf == NULL)
    return GA_MEMORY_ERROR;

  err = GpuArray_read(buf, s, a);
  if (err != GA_NO_ERROR) {
    free(buf);
    return err;
  }

  p = buf;
  k = 0;
  while (s) {
    fprintf(fd, "[%" SPREFIX "u] = ", k);
    switch (a->typecode) {
    case GA_UINT:
      fprintf(fd, "%u", *(unsigned int *)p);
      break;
    case GA_SSIZE:
      fprintf(fd, "%" SPREFIX "d", *(ssize_t *)p);
      break;
    default:
      free(buf);
      return GA_UNSUPPORTED_ERROR;
    }
    s -= gpuarray_get_elsize(a->typecode);
    p += gpuarray_get_elsize(a->typecode);
    k++;
    fprintf(fd, "\n");
  }
  free(buf);
  return GA_NO_ERROR;
}

int GpuArray_is_c_contiguous(const GpuArray *a) {
  size_t size = GpuArray_ITEMSIZE(a);
  int i;

  for (i = a->nd - 1; i >= 0; i--) {
    if (a->dimensions[i] != 1 && a->strides[i] != size)
      return 0;
    // We suppose that overflow will not happen since data has to fit in memory
    size *= a->dimensions[i];
  }
  return 1;
}

int GpuArray_is_f_contiguous(const GpuArray *a) {
  size_t size = GpuArray_ITEMSIZE(a);
  unsigned int i;

  for (i = 0; i < a->nd; i++) {
    if (a->dimensions[i] != 1 && a->strides[i] != size) return 0;
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
