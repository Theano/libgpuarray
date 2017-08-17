#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "private.h"
#include "gpuarray/config.h"
#include "gpuarray/array.h"
#include "gpuarray/error.h"
#include "gpuarray/kernel.h"
#include "gpuarray/util.h"

#include "util/error.h"
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
  gpucontext *ctx = GpuArray_context(dst);
  GpuElemwise *k = NULL;
  void *args[2];

  if (ctx != GpuArray_context(src))
    return error_set(ctx->err, GA_INVALID_ERROR, "src and dst context differ");

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
    k = GpuElemwise_new(ctx, "", "dst = src", 2, gargs, 0, GE_CONVERT_F16);
    if (k == NULL)
      return ctx->err->code;
    aa = memdup(&a, sizeof(a));
    if (aa == NULL) {
      GpuElemwise_free(k);
      return error_sys(ctx->err, "memdup");
    }
    if (ctx->extcopy_cache == NULL)
      ctx->extcopy_cache = cache_twoq(4, 8, 8, 2, extcopy_eq, extcopy_hash,
                                      extcopy_free,
                                      (cache_freev_fn)GpuElemwise_free,
                                      ctx->err);
    if (ctx->extcopy_cache == NULL)
      return ctx->err->code;
    if (cache_add(ctx->extcopy_cache, aa, k) != 0)
      return error_set(ctx->err, GA_MISC_ERROR,
                       "Could not store GpuElemwise copy kernel in context cache");
  }
  args[0] = (void *)src;
  args[1] = (void *)dst;
  return GpuElemwise_call(k, args, GE_BROADCAST);
}

/* Value below which a size_t multiplication will never overflow. */
#define MUL_NO_OVERFLOW (1ULL << (sizeof(size_t) * 4))

void GpuArray_fix_flags(GpuArray *a) {
  /* Only keep the writable flag */
  a->flags &= GA_WRITEABLE;
  /* Set the other flags if applicable */
  if (GpuArray_is_c_contiguous(a)) a->flags |= GA_C_CONTIGUOUS;
  if (GpuArray_is_f_contiguous(a)) a->flags |= GA_F_CONTIGUOUS;
  if (GpuArray_is_aligned(a)) a->flags |= GA_ALIGNED;
}

int GpuArray_empty(GpuArray *a, gpucontext *ctx, int typecode,
                   unsigned int nd, const size_t *dims, ga_order ord) {
  size_t size = gpuarray_get_elsize(typecode);
  unsigned int i;
  int res = GA_NO_ERROR;

  if (typecode == GA_SIZE || typecode == GA_SSIZE)
    return error_set(ctx->err, GA_VALUE_ERROR, "Cannot create array with size type");

  if (ord == GA_ANY_ORDER)
    ord = GA_C_ORDER;

  if (ord != GA_C_ORDER && ord != GA_F_ORDER)
    return error_set(ctx->err, GA_VALUE_ERROR, "Invalid order");

  for (i = 0; i < nd; i++) {
    size_t d = dims[i];
    /* Check for overflow */
    if ((d >= MUL_NO_OVERFLOW || size >= MUL_NO_OVERFLOW) &&
        d > 0 && SIZE_MAX / d < size)
      return error_set(ctx->err, GA_XLARGE_ERROR, "Total array size greater than addressable space");
    size *= d;
  }

  /* We add a offset of 64 to all arrays in DEBUG to help catch errors. */
#ifdef DEBUG
  assert(SIZE_MAX - size > 64);
  size += 64;
#endif

  a->data = gpudata_alloc(ctx, size, NULL, 0, &res);
  if (a->data == NULL) return ctx->err->code;
  a->nd = nd;
#ifdef DEBUG
  a->offset = 64;
#else
  a->offset = 0;
#endif
  a->typecode = typecode;
  a->dimensions = calloc(nd, sizeof(size_t));
  a->strides = calloc(nd, sizeof(ssize_t));
  /* F/C distinction comes later */
  a->flags = GA_BEHAVED;
  if (a->dimensions == NULL || a->strides == NULL) {
    GpuArray_clear(a);
    return error_sys(ctx->err, "calloc");
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
  gpucontext *ctx = gpudata_context(data);

  if (typecode == GA_SIZE || typecode == GA_SSIZE)
    return error_set(ctx->err, GA_VALUE_ERROR, "Cannot create array with size type");

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
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");
  }
  memcpy(a->dimensions, dims, nd*sizeof(size_t));
  memcpy(a->strides, strides, nd*sizeof(ssize_t));

  GpuArray_fix_flags(a);

  return GA_NO_ERROR;
}

int GpuArray_view(GpuArray *v, const GpuArray *a) {
  gpucontext *ctx = GpuArray_context(a);
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
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");
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
  gpucontext *ctx = GpuArray_context(a);
  unsigned int i, new_i;
  unsigned int new_nd = a->nd;
  size_t *newdims;
  ssize_t *newstrs;
  size_t new_offset = a->offset;

  if ((starts == NULL) || (stops == NULL) || (steps == NULL))
    return error_set(ctx->err, GA_VALUE_ERROR, "Invalid slice (contains NULL)");

  for (i = 0; i < a->nd; i++) {
    if (steps[i] == 0) new_nd -= 1;
  }
  newdims = calloc(new_nd, sizeof(size_t));
  newstrs = calloc(new_nd, sizeof(ssize_t));
  if (newdims == NULL || newstrs == NULL) {
    free(newdims);
    free(newstrs);
    return error_sys(ctx->err, "calloc");
  }

  new_i = 0;
  for (i = 0; i < a->nd; i++) {
    if (starts[i] < -1 || (starts[i] > 0 &&
                           (size_t)starts[i] > a->dimensions[i])) {
      free(newdims);
      free(newstrs);
      return error_fmt(ctx->err, GA_VALUE_ERROR,
                       "Invalid slice value: slice(%lld, %lld, %lld) when "
                       "indexing array on dimension %u of length %lld",
                       starts[i], stops[i], steps[i], i, a->dimensions[i]);
    }
    if (steps[i] == 0 &&
        (starts[i] == -1 || (size_t)starts[i] >= a->dimensions[i])) {
      free(newdims);
      free(newstrs);
      return error_fmt(ctx->err, GA_VALUE_ERROR,
                       "Invalid slice value: slice(%lld, %lld, %lld) when "
                       "indexing array on dimension %u of length %lld",
                       starts[i], stops[i], steps[i], i, a->dimensions[i]);
    }
    new_offset += starts[i] * a->strides[i];
    if (steps[i] != 0) {
      if ((stops[i] < -1 || (stops[i] > 0 &&
                             (size_t)stops[i] > a->dimensions[i])) ||
          (stops[i]-starts[i])/steps[i] < 0) {
        free(newdims);
        free(newstrs);
        return error_fmt(ctx->err, GA_VALUE_ERROR,
                         "Invalid slice value: slice(%lld, %lld, %lld) when "
                         "indexing array on dimension %u of length %lld",
                         starts[i], stops[i], steps[i], i, a->dimensions[i]);
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
  GpuArray_fix_flags(a);

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
  char *sz, *ssz;
  unsigned int i, i2;
  unsigned int nargs, apos;
  int flags = 0;
  int res;

  nargs = 9 + 2 * v->nd;

  atypes = calloc(nargs, sizeof(int));
  if (atypes == NULL)
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");

  if (addr32) {
    sz = "ga_uint";
    ssz = "ga_int";
  } else {
    sz = "ga_size";
    ssz = "ga_ssize";
  }

  apos = 0;
  strb_appendf(&sb, "#include \"cluda.h\"\n"
               "KERNEL void take1(GLOBAL_MEM %s *r, ga_size r_off, "
               "GLOBAL_MEM const %s *v, ga_size v_off,",
               gpuarray_get_type(a->typecode)->cluda_name,
               gpuarray_get_type(v->typecode)->cluda_name);
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_SIZE;
  for (i = 0; i < v->nd; i++) {
    strb_appendf(&sb, " ga_ssize s%u, ga_size d%u,", i, i);
    atypes[apos++] = GA_SSIZE;
    atypes[apos++] = GA_SIZE;
  }
  strb_appendf(&sb, " GLOBAL_MEM const %s *ind, ga_size i_off, "
               "ga_size n0, ga_size n1, GLOBAL_MEM int* err) {\n",
               gpuarray_get_type(ind->typecode)->cluda_name);
  atypes[apos++] = GA_BUFFER;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_SIZE;
  atypes[apos++] = GA_BUFFER;
  assert(apos == nargs);
  strb_appendf(&sb, "  const %s idx0 = LDIM_0 * GID_0 + LID_0;\n"
               "  const %s numThreads0 = LDIM_0 * GDIM_0;\n"
               "  const %s idx1 = LDIM_1 * GID_1 + LID_1;\n"
               "  const %s numThreads1 = LDIM_1 * GDIM_1;\n"
               "  %s i0, i1;\n", sz, sz, sz, sz, sz);
  strb_appends(&sb, "  if (idx0 >= n0 || idx1 >= n1) return;\n");
  strb_appendf(&sb, "  r = (GLOBAL_MEM %s *)(((GLOBAL_MEM char *)r) + r_off);\n"
               "  ind = (GLOBAL_MEM %s *)(((GLOBAL_MEM char *)ind) + i_off);\n",
               gpuarray_get_type(a->typecode)->cluda_name,
               gpuarray_get_type(ind->typecode)->cluda_name);
  strb_appendf(&sb, "  for (i0 = idx0; i0 < n0; i0 += numThreads0) {\n"
               "    %s ii0 = ind[i0];\n"
               "    %s pos0 = v_off;\n"
               "    if (ii0 < 0) ii0 += d0;\n"
               "    if ((ii0 < 0) || (ii0 >= (%s)d0)) {\n"
               "      *err = -1;\n"
               "      continue;\n"
               "    }\n"
               "    pos0 += ii0 * (%s)s0;\n"
               "    for (i1 = idx1; i1 < n1; i1 += numThreads1) {\n"
               "      %s p = pos0;\n", ssz, sz, ssz, sz, sz);
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
    res = error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");
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
  gpucontext *ctx = GpuArray_context(a);
  size_t n[2], ls[2] = {0, 0}, gs[2] = {0, 0};
  size_t pl;
  gpudata *errbuf;
#if DEBUG
  char *errstr = NULL;
#endif
  GpuKernel k;
  unsigned int j;
  unsigned int argp;
  int err, kerr = 0;
  int addr32 = 0;

  if (!GpuArray_ISWRITEABLE(a))
    return error_set(ctx->err, GA_VALUE_ERROR, "Destination array not writeable");

  if (!GpuArray_ISALIGNED(a) || !GpuArray_ISALIGNED(v) ||
      !GpuArray_ISALIGNED(i))
    return error_fmt(ctx->err, GA_UNALIGNED_ERROR,
                     "Not all arrays are aligned: a (%d), b (%d), i (%d)",
                     GpuArray_ISALIGNED(a), GpuArray_ISALIGNED(v), GpuArray_ISALIGNED(i));

  /* a and i have to be C contiguous */
  if (!GpuArray_IS_C_CONTIGUOUS(a))
    return error_set(ctx->err, GA_INVALID_ERROR, "Destination array (a) not C-contiguous");
  if (!GpuArray_IS_C_CONTIGUOUS(i))
    return error_set(ctx->err, GA_INVALID_ERROR, "Index array (i) not C-contiguous");

  /* Check that the dimensions match namely a[0] == i[0] and a[>0] == v[>0] */
  if (v->nd == 0 || a->nd == 0 || i->nd != 1 || a->nd != v->nd)
    return error_fmt(ctx->err, GA_INVALID_ERROR, "Dimension mismatch. "
                     "v->nd = %llu, a->nd = %llu, i->nd = %llu",
                     v->nd, a->nd, i->nd);
  if (a->dimensions[0] != i->dimensions[0])
    return error_fmt(ctx->err, GA_INVALID_ERROR, "Dimension mismatch. "
                     "a->dimensions[0] = %llu, i->dimensions[0] = %llu",
                     a->dimensions[0], i->dimensions[0]);

  n[0] = i->dimensions[0];
  n[1] = 1;

  for (j = 1; j < v->nd; j++) {
    if (a->dimensions[j] != v->dimensions[j])
      return error_fmt(ctx->err, GA_INVALID_ERROR, "Dimension mismatch. "
                       "a->dimensions[%llu] = %llu, i->dimensions[%llu] = %llu",
                       j, a->dimensions[j], j, i->dimensions[j]);
    n[1] *= v->dimensions[j];
  }

  if (n[0] * n[1] < SADDR32_MAX) {
    addr32 = 1;
  }

  err = gpudata_property(v->data, GA_CTX_PROP_ERRBUF, &errbuf);
  if (err != GA_NO_ERROR)
    return err;

  err = gen_take1_kernel(&k, ctx,
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

  err = GpuKernel_sched(&k, n[0]*n[1], &gs[1], &ls[1]);
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
    gs[0] = 1;
  } else {
    gs[0] = gs[1];
    gs[1] = 1;
  }

  argp = 0;
  GpuKernel_setarg(&k, argp++, a->data);
  GpuKernel_setarg(&k, argp++, (void *)&a->offset);
  GpuKernel_setarg(&k, argp++, v->data);
  /* The cast is to avoid a warning about const */
  GpuKernel_setarg(&k, argp++, (void *)&v->offset);
  for (j = 0; j < v->nd; j++) {
    GpuKernel_setarg(&k, argp++, &v->strides[j]);
    GpuKernel_setarg(&k, argp++, &v->dimensions[j]);
  }
  GpuKernel_setarg(&k, argp++, i->data);
  GpuKernel_setarg(&k, argp++, (void *)&i->offset);
  GpuKernel_setarg(&k, argp++, &n[0]);
  GpuKernel_setarg(&k, argp++, &n[1]);
  GpuKernel_setarg(&k, argp++, errbuf);

  err = GpuKernel_call(&k, 2, gs, ls, 0, NULL);
  if (check_error && err == GA_NO_ERROR) {
    err = gpudata_read(&kerr, errbuf, 0, sizeof(int));
    if (err == GA_NO_ERROR && kerr != 0) {
      err = error_set(ctx->err, GA_VALUE_ERROR, "Index out of bounds");
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
  gpucontext *ctx = GpuArray_context(a);
  GpuArray tv;
  size_t sz;
  ssize_t *strs;
  unsigned int i, off;
  int err = GA_NO_ERROR;
  int simple_move = 1;

  if (a->nd < v->nd)
    return error_fmt(ctx->err, GA_VALUE_ERROR, "Dimension error. "
                     "a->nd = %llu, v->nd = %llu", a->nd, v->nd);

  if (!GpuArray_ISWRITEABLE(a))
    return error_set(ctx->err, GA_VALUE_ERROR, "Destination array not writable");
  if (!GpuArray_ISALIGNED(v) || !GpuArray_ISALIGNED(a))
    return error_set(ctx->err, GA_UNALIGNED_ERROR, "One of the inputs is unaligned");

  off = a->nd - v->nd;

  for (i = 0; i < v->nd; i++) {
    if (v->dimensions[i] != a->dimensions[i+off]) {
      if (v->dimensions[i] != 1)
        return error_fmt(ctx->err, GA_VALUE_ERROR, "Shape error. "
                         "v->dimensions[%u] = %llu, a->dimesions[%u + %u] = %llu",
                         i, v->dimensions[i], i, off, a->dimensions[i + off]);
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
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");

  for (i = off; i < a->nd; i++) {
    if (v->dimensions[i-off] == a->dimensions[i]) {
      strs[i] = v->strides[i-off];
    }
  }

  memcpy(&tv, v, sizeof(GpuArray));
  tv.nd = a->nd;
  tv.dimensions = a->dimensions;
  tv.strides = strs;
  if (tv.nd != 0)
    GpuArray_fix_flags(&tv);
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
    err = GpuArray_copy(res, a, ord);
    if (err != GA_NO_ERROR) return err;
    err = GpuArray_reshape_inplace(res, nd, newdims, ord);
  }
  if (err != GA_NO_ERROR) GpuArray_clear(res);
  return err;
}

int GpuArray_reshape_inplace(GpuArray *a, unsigned int nd,
                             const size_t *newdims, ga_order ord) {
  gpucontext *ctx = GpuArray_context(a);
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
      return error_set(ctx->err, GA_XLARGE_ERROR, "Output array size greater than addressable space");
    newsize *= d;
  }

  if (newsize != oldsize) return error_set(ctx->err, GA_INVALID_ERROR, "New shape differs in total size");

  /* If the source and desired layouts are the same, then just copy
     strides and dimensions */
  if ((ord == GA_C_ORDER && GpuArray_CHKFLAGS(a, GA_C_CONTIGUOUS)) ||
      (ord == GA_F_ORDER && GpuArray_CHKFLAGS(a, GA_F_CONTIGUOUS))) {
    goto do_final_copy;
  }

  newstrides = calloc(nd, sizeof(ssize_t));
  if (newstrides == NULL)
    return error_sys(ctx->err, "calloc");

  if (newsize != 0) {
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
          if (a->strides[ok+1] != (ssize_t)a->dimensions[ok]*a->strides[ok])
            goto need_copy;
        } else {
          if (a->strides[ok] != (ssize_t)a->dimensions[ok+1]*a->strides[ok+1])
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
    return error_sys(ctx->err, "calloc");
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
  return error_set(ctx->err, GA_COPY_ERROR, "Copy is needed but disallowed by parameters");

 do_final_copy:
  tmpdims = calloc(nd, sizeof(size_t));
  newstrides = calloc(nd, sizeof(ssize_t));
  if (tmpdims == NULL || newstrides == NULL) {
    free(tmpdims);
    free(newstrides);
    return error_sys(ctx->err, "calloc");
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
  GpuArray_fix_flags(a);
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
  gpucontext *ctx = GpuArray_context(a);
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
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");
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
          return error_fmt(ctx->err, GA_VALUE_ERROR,
                           "Repeated axes in transpose: new_axes[%u] == new_axes[%u] == %u",
                           i, k, j);
        }
    }
    newdims[i] = a->dimensions[j];
    newstrs[i] = a->strides[j];
  }

  free(a->dimensions);
  free(a->strides);
  a->dimensions = newdims;
  a->strides = newstrs;

  GpuArray_fix_flags(a);

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
  gpucontext *ctx = GpuArray_context(dst);
  size_t sz;
  unsigned int i;
  if (!GpuArray_ISWRITEABLE(dst))
    return error_set(ctx->err, GA_VALUE_ERROR, "Destination array (dst) not writeable");
  if (!GpuArray_ISALIGNED(src))
    return error_set(ctx->err, GA_UNALIGNED_ERROR, "Source array (src) not aligned");
  if (!GpuArray_ISALIGNED(dst))
    return error_set(ctx->err, GA_UNALIGNED_ERROR, "Destination array (dst) not aligned");
  if (src->nd != dst->nd)
    return error_fmt(ctx->err, GA_VALUE_ERROR,
                     "Dimension mismatch. src->nd = %llu, dst->nd = %llu",
                     src->nd, dst->nd);
  for (i = 0; i < src->nd; i++) {
    if (src->dimensions[i] != dst->dimensions[i])
      return error_fmt(ctx->err, GA_VALUE_ERROR,
                       "Dimension mismatch. src->dimensions[%u] = %llu, dst->dimensions[%u] = %llu",
                       i, src->dimensions[i], i, dst->dimensions[i]);
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
  gpucontext *ctx = GpuArray_context(dst);
  if (!GpuArray_ISWRITEABLE(dst))
    return error_set(ctx->err, GA_VALUE_ERROR, "Destination array (dst) not writeable");
  if (!GpuArray_ISONESEGMENT(dst))
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Destination array (dst) not one segment");
  return gpudata_write(dst->data, dst->offset, src, src_sz);
}

int GpuArray_read(void *dst, size_t dst_sz, const GpuArray *src) {
  gpucontext *ctx = GpuArray_context(src);
  if (!GpuArray_ISONESEGMENT(src))
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Array (src) not one segment");
  return gpudata_read(dst, src->data, src->offset, dst_sz);
}

int GpuArray_memset(GpuArray *a, int data) {
  gpucontext *ctx = GpuArray_context(a);
  if (!GpuArray_ISONESEGMENT(a))
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Array (a) not one segment");
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
  gpucontext *ctx = GpuArray_context(res);
  size_t sz;
  unsigned int i;

  if (!GpuArray_ISONESEGMENT(res))
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Array (res) not one segment");
  if (!GpuArray_ISONESEGMENT(a))
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Array (a) not one segment");

  if (res->typecode != a->typecode)
    return error_set(ctx->err, GA_UNSUPPORTED_ERROR, "typecode mismatch");

  sz = gpuarray_get_elsize(a->typecode);
  for (i = 0; i < a->nd; i++) sz *= a->dimensions[i];

 return gpudata_transfer(res->data, res->offset, a->data, a->offset, sz);
}

int GpuArray_split(GpuArray **rs, const GpuArray *a, size_t n, size_t *p,
                   unsigned int axis) {
  gpucontext *ctx = GpuArray_context(a);
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
    return error_sys(ctx->err, "calloc");
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
  gpucontext *ctx = GpuArray_context(as[0]);
  size_t *dims, *res_dims;
  size_t i, res_off;
  unsigned int p;
  int res_flags;
  int err = GA_NO_ERROR;

  if (axis >= as[0]->nd)
    return error_fmt(ctx->err, GA_VALUE_ERROR, "Invalid axis. "
                     "axis = %u, as[0]->nd = %llu", axis, as[0]->nd);

  dims = calloc(as[0]->nd, sizeof(size_t));
  if (dims == NULL)
    return error_fmt(ctx->err, GA_MEMORY_ERROR, "Out of memory");

  for (p = 0; p < as[0]->nd; p++) {
    dims[p] = as[0]->dimensions[p];
  }

  if (!GpuArray_ISALIGNED(as[0])) {
    err = error_set(ctx->err, GA_UNALIGNED_ERROR, "Unaligned array (as[0]).");
    goto afterloop;
  }

  for (i = 1; i < n; i++) {
    if (!GpuArray_ISALIGNED(as[i])) {
      err = error_fmt(ctx->err, GA_UNALIGNED_ERROR, "Unaligned array (as[%llu]).", i);
      goto afterloop;
    }
    if (as[i]->nd != as[0]->nd) {
      err = error_fmt(ctx->err, GA_VALUE_ERROR, "Shape mismatch. "
                      "as[%llu]->nd = %llu, as[0]->nd = %llu",
                      i, as[i]->nd, as[0]->nd);
      goto afterloop;
    }
    for (p = 0; p < as[0]->nd; p++) {
      if (p != axis && dims[p] != as[i]->dimensions[p]) {
        err = error_fmt(ctx->err, GA_VALUE_ERROR, "Dimension mismatch. "
                        "as[%llu]->dimensions[%u] = %llu, as[0]->dimensions[%u] = %llu",
                        i, p, as[i]->dimensions[p], p, dims[p]);
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
  for (i = 0; i < n; i++) {
    r->dimensions = as[i]->dimensions;
    GpuArray_fix_flags(r);
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

  fprintf(fd, "GpuArray <%p, data: %p (%p)> nd=%d\n",
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
  if (!GpuArray_is_c_contiguous(a) && ISSET(a->flags, GA_C_CONTIGUOUS))
    fputc('!', fd);
  PRINTFLAG(GA_F_CONTIGUOUS);
  if (!GpuArray_is_f_contiguous(a) && ISSET(a->flags, GA_F_CONTIGUOUS))
    fputc('!', fd);
  PRINTFLAG(GA_ALIGNED);
  PRINTFLAG(GA_WRITEABLE);
#undef PRINTFLAG
  fputc('\n', fd);
}

int GpuArray_fdump(FILE *fd, const GpuArray *a) {
  gpucontext *ctx = GpuArray_context(a);
  char *buf, *p;
  size_t s = GpuArray_ITEMSIZE(a);
  size_t k;
  unsigned int i;
  int err;

  for (i = 0; i < a->nd; i++)
    s *= a->dimensions[i];

  buf = malloc(s);
  if (buf == NULL)
    return error_set(ctx->err, GA_MEMORY_ERROR, "Out of memory");

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
    case GA_LONG:
      fprintf(fd, "%lld", (long long)*(int64_t *)p);
      break;
    case GA_FLOAT:
      fprintf(fd, "%f", *(float *)p);
      break;
    case GA_SSIZE:
      fprintf(fd, "%" SPREFIX "d", *(ssize_t *)p);
      break;
    default:
      free(buf);
      fprintf(fd, "<unsupported data type %d>\n", a->typecode);
      return error_fmt(ctx->err, GA_UNSUPPORTED_ERROR, "Unsupported data type for dump: %d", a->typecode);
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
    if (a->strides[i] != (ssize_t)size && a->dimensions[i] != 1) return 0;
    // We suppose that overflow will not happen since data has to fit in memory
    size *= a->dimensions[i];
  }
  return 1;
}

int GpuArray_is_f_contiguous(const GpuArray *a) {
  size_t size = GpuArray_ITEMSIZE(a);
  unsigned int i;

  for (i = 0; i < a->nd; i++) {
    if (a->strides[i] != (ssize_t)size && a->dimensions[i] != 1) return 0;
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
