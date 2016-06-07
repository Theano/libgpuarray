#include <assert.h>

#include <gpuarray/elemwise.h>
#include <gpuarray/array.h>
#include <gpuarray/error.h>
#include <gpuarray/kernel.h>
#include <gpuarray/util.h>

#include "private.h"
#include "util/strb.h"

struct _GpuElemwise {
  const char *expr; /* Expression code (to be able to build kernels on-demand) */
  const char *preamble; /* Preamble code */
  gpuelemwise_arg *args; /* Argument descriptors */
  GpuKernel k_contig; /* Contiguous kernel */
  GpuKernel *k_basic; /* Normal basic kernels */
  GpuKernel *k_basic_32; /* 32-bit address basic kernels */
  size_t *dims; /* Preallocated shape buffer for dimension collapsing */
  ssize_t **strides; /* Preallocated strides buffer for dimension collapsing */
  unsigned int nd; /* Current maximum number of dimensions allocated */
  unsigned int n; /* Number of arguments */
  unsigned int narray; /* Number of array arguments */
  int flags; /* Flags for the operation (none at the moment */
};

#define GEN_ADDR32      0x1
#define GEN_CONVERT_F16 0x2

/* This makes sure we have the same value for those flags since we use some shortcuts */
STATIC_ASSERT(GEN_CONVERT_F16 == GE_CONVERT_F16, same_flags_value_elem1);

#define is_array(a) (ISCLR((a).flags, GE_SCALAR))

static inline int k_initialized(GpuKernel *k) {
  return k->k != NULL;
}

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode)->cluda_name;
}

/* dst has to be zero-initialized on entry */
static int copy_arg(gpuelemwise_arg *dst, gpuelemwise_arg *src) {
  dst->name = strdup(src->name);
  if (dst->name == NULL)
    return -1;

  dst->typecode = src->typecode;
  dst->flags = src->flags;

  return 0;
}

static void clear_arg(gpuelemwise_arg *a) {
  free((void *)a->name);
  a->name = NULL;
}

static gpuelemwise_arg *copy_args(unsigned int n, gpuelemwise_arg *a) {
  gpuelemwise_arg *res = calloc(n, sizeof(gpuelemwise_arg));
  unsigned int i;

  if (res == NULL) return NULL;

  for (i = 0; i < n; i++)
    if (copy_arg(&res[i], &a[i]) != 0)
      goto bail;

  return res;
 bail:
  for (; i > 0; i--) {
    clear_arg(&res[i]);
  }
  return NULL;
}

static void free_args(unsigned int n, gpuelemwise_arg *args) {
  unsigned int i;

  if (args != NULL)
    for (i = 0; i < n; i++)
      clear_arg(&args[i]);
  free(args);
}

#define MUL_NO_OVERFLOW ((size_t)1 << (sizeof(size_t) * 4))

static int reallocaz(void **p, size_t elsz, size_t old, size_t new) {
  char *res;

  assert(old <= new);

  if ((new >= MUL_NO_OVERFLOW || elsz >= MUL_NO_OVERFLOW) &&
      new > 0 && SIZE_MAX / new < elsz) {
    return 1;
  }
  res = realloc(*p, elsz*new);
  if (res == NULL) return 1;
  memset(res + (elsz*old), 0, elsz*(new-old));
  *p = (void *)res;
  return 0;
}

static int ge_grow(GpuElemwise *ge, unsigned int nd) {
  unsigned int i;

  assert(nd > ge->nd);

  if (reallocaz((void **)&ge->k_basic, sizeof(GpuKernel), ge->nd, nd) ||
      reallocaz((void **)&ge->k_basic_32, sizeof(GpuKernel), ge->nd, nd) ||
      reallocaz((void **)&ge->dims, sizeof(size_t), ge->nd, nd))
    return 1;
  for (i = 0; i < ge->narray; i++) {
    if (reallocaz((void **)&ge->strides[i], sizeof(ssize_t), ge->nd, nd))
      return 1;
  }
  ge->nd = nd;
  return 0;
}

static int gen_elemwise_basic_kernel(GpuKernel *k, gpucontext *ctx,
                                     char **err_str,
                                     const char *preamble,
                                     const char *expr,
                                     unsigned int nd, /* Number of dims */
                                     unsigned int n, /* Length of args */
                                     gpuelemwise_arg *args,
                                     int gen_flags) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *ktypes;
  size_t p;
  char *size = "ga_size", *ssize = "ga_ssize";
  int flags = GA_USE_CLUDA;
  int res;

  if (ISSET(gen_flags, GEN_ADDR32)) {
    size = "ga_uint";
    ssize = "ga_int";
  }

  flags |= gpuarray_type_flagsa(n, args);

  p = 1 + nd;
  for (j = 0; j < n; j++) {
    p += ISSET(args[j].flags, GE_SCALAR) ? 1 : (2 + nd);
  }

  ktypes = calloc(p, sizeof(int));
  if (ktypes == NULL)
    return GA_MEMORY_ERROR;

  p = 0;

  if (preamble)
    strb_appends(&sb, preamble);
  strb_appends(&sb, "\nKERNEL void elem(const ga_size n, ");
  ktypes[p++] = GA_SIZE;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb, "const ga_size dim%u, ", i);
    ktypes[p++] = GA_SIZE;
  }
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "GLOBAL_MEM %s *%s_data, const ga_size %s_offset%s",
                   ctype(args[j].typecode), args[j].name, args[j].name,
                   nd == 0 ? "" : ", ");
      ktypes[p++] = GA_BUFFER;
      ktypes[p++] = GA_SIZE;

      for (i = 0; i < nd; i++) {
        strb_appendf(&sb, "const ga_ssize %s_str_%u%s", args[j].name, i,
                     (i == (nd - 1)) ? "": ", ");
        ktypes[p++] = GA_SSIZE;
      }
    } else {
      strb_appendf(&sb, "%s %s", ctype(args[i].typecode), args[j].name);
      ktypes[p++] = args[j].typecode;
    }
    if (j != (n - 1)) strb_appends(&sb, ", ");
  }
  strb_appendf(&sb, ") {\n"
               "const %s idx = LDIM_0 * GID_0 + LID_0;\n"
               "const %s numThreads = LDIM_0 * GDIM_0;\n"
               "%s i;\n", size, size, size);

  strb_appends(&sb, "for(i = idx; i < n; i += numThreads) {\n");
  if (nd > 0)
    strb_appendf(&sb, "%s ii = i;\n%s pos;\n", size, size);
  for (j = 0; j < n; j++) {
    if (is_array(args[j]))
      strb_appendf(&sb, "%s %s_p = %s_offset;\n",
                   size, args[j].name, args[j].name);
  }
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (i > 0)
      strb_appendf(&sb, "pos = ii %% (%s)dim%u;\nii = ii / (%s)dim%u;\n", size, i, size, i);
    else
      strb_appends(&sb, "pos = ii;\n");
    for (j = 0; j < n; j++) {
      if (is_array(args[j]))
        strb_appendf(&sb, "%s_p += pos * (%s)%s_str_%u;\n", args[j].name,
                     ssize, args[j].name, i);
    }
  }
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "%s %s;", ctype(ISSET(gen_flags, GEN_CONVERT_F16) && args[j].typecode == GA_HALF ?
                                        GA_FLOAT : args[j].typecode), args[j].name);
      if (ISSET(args[j].flags, GE_READ)) {
        if (args[j].typecode == GA_HALF && ISSET(gen_flags, GEN_CONVERT_F16)) {
          strb_appendf(&sb, "%s = load_half((GLOBAL_MEM ga_half *)(((GLOBAL_MEM char *)%s_data) + %s_p));\n",
                       args[j].name, args[j].name, args[j].name);
        } else {
          strb_appendf(&sb, "%s = *(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)%s_data) + %s_p);\n",
                       args[j].name, ctype(args[j].typecode), args[j].name, args[j].name);
        }
      }
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j]) && ISSET(args[j].flags, GE_WRITE)) {
      if (args[j].typecode == GA_HALF && ISSET(gen_flags, GEN_CONVERT_F16)) {
        strb_appendf(&sb, "store_half((GLOBAL_MEM ga_half *)(((GLOBAL_MEM char *)%s_data) + %s_p), %s);\n",
                     args[j].name, args[j].name, args[j].name);
      } else {
        strb_appendf(&sb, "*(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)%s_data) + %s_p) = %s;\n",
                     ctype(args[j].typecode), args[j].name, args[j].name, args[j].name);
      }
    }
  }
  strb_appends(&sb, "}\n}\n");
  if (strb_error(&sb)) {
    res = GA_MEMORY_ERROR;
    goto bail;
  }

  res = GpuKernel_init(k, ctx, 1, (const char **)&sb.s, &sb.l, "elem",
                       p, ktypes, flags, err_str);
 bail:
  free(ktypes);
  strb_clear(&sb);
  return res;
}

static ssize_t **strides_array(unsigned int num, unsigned int nd) {
  ssize_t **res = calloc(num, sizeof(ssize_t *));
  unsigned int i;

  if (res == NULL) return NULL;
  for (i = 0; i < num; i++) {
    res[i] = calloc(nd, sizeof(ssize_t));
    if (res[i] == NULL)
      goto bail;
  }

  return res;

 bail:
  for (i = 0; i < num; i++)
    free(res[i]);
  free(res);
  return NULL;
}

static int check_basic(GpuElemwise *ge, void **args, int flags,
                       size_t *_n, unsigned int *_nd, size_t **_dims,
                       ssize_t ***_strides, int *_call32) {
  size_t n;
  GpuArray *a = NULL, *v;
  unsigned int i, j, p, num_arrays = 0, nd = 0, nnd;
  int call32 = 1;

  /* Go through the list and grab some info */
  for (i = 0; i < ge->n; i++) {
    if (is_array(ge->args[i])) {
      num_arrays++;
      if (a == NULL) {
        a = (GpuArray *)args[i];
        nd = a->nd;
      }
      if (((GpuArray *)args[i])->nd != nd)
        return GA_VALUE_ERROR;
    }
  }

  if (a == NULL)
    return GA_VALUE_ERROR;

  /* Check if we need to grow the internal buffers */
  if (nd > ge->nd) {
    nnd = ge->nd * 2;
    while (nd > nnd) nnd *= 2;
    if (ge_grow(ge, nnd))
      return GA_MEMORY_ERROR;
  }

  /* Now we know that all array arguments have the same number of
     dimensions */

  /* And copy their initial values in */
  memcpy(ge->dims, a->dimensions, nd*sizeof(size_t));
  p = 0;
  for (i = 0; i < ge->n; i++) {
    if (is_array(ge->args[i])) {
      memcpy(ge->strides[p], ((GpuArray *)args[i])->strides, nd*sizeof(ssize_t));
      p++;
    }
  }

  /* Check that all arrays are the same size (or broadcast-compatible
     if GE_BROADCAST).  Also compute the total size and adjust strides
     of broadcastable dimensions.

     Basically for each dimension go over all the arguments and make
     sure that the dimension size matches. */
  n = 1;
  for (j = 0; j < nd; j++) {
    p = 0;
    for (i = 0; i < ge->n; i++) {
      if (is_array(ge->args[i])) {
        v = (GpuArray *)args[i];
        if (ge->dims[j] != v->dimensions[j]) {
          if (ISCLR(flags, GE_BROADCAST)) {
            return GA_VALUE_ERROR;
          }
          /* GE_BROADCAST is set */
          if (ge->dims[j] == 1) {
            ge->dims[j] = v->dimensions[j];
          } else {
            if (v->dimensions[j] != 1) {
              return GA_VALUE_ERROR;
            }
          }
        }
        /* If the dimension is 1 set the strides to 0 regardless since
           it won't change anything in the non-broadcast case. */
        if (v->dimensions[j] == 1) {
          ge->strides[p][j] = 0;
        }
        call32 &= v->offset < ADDR32_MAX;
        call32 &= (SADDR32_MIN < ge->strides[p][j] &&
                   ge->strides[p][j] < SADDR32_MAX);
        p++;
      } /* is_array() */
    } /* for each arg */
    /* We have the final value in dims[j] */
    n *= ge->dims[j];
  } /* for each dim */

  call32 &= n < ADDR32_MAX;

  if (ISCLR(flags, GE_NOCOLLAPSE) && nd > 1) {
    gpuarray_elemwise_collapse(num_arrays, &nd, ge->dims, ge->strides);
  }

  *_n = n;
  *_nd = nd;
  *_dims = ge->dims;
  *_strides = ge->strides;
  *_call32 = call32;

  return GA_NO_ERROR;
}

static int call_basic(GpuElemwise *ge, void **args, size_t n, unsigned int nd,
                      size_t *dims, ssize_t **strs, int call32) {
  GpuKernel *k;
  size_t ls = 0, gs = 0;
  unsigned int p = 0, i, j;
  int err;

  if (nd == 0) return GA_VALUE_ERROR;

  if (call32)
    k = &ge->k_basic_32[nd-1];
  else
    k = &ge->k_basic[nd-1];

  if (!k_initialized(k)) {
    err = gen_elemwise_basic_kernel(k, GpuKernel_context(&ge->k_contig), NULL,
                                    ge->preamble, ge->expr, nd, ge->n,
                                    ge->args, ((call32 ? GEN_ADDR32 : 0) |
                                               (ge->flags & GE_CONVERT_F16)));
    if (err != GA_NO_ERROR)
      return err;
  }

  err = GpuKernel_setarg(k, p++, &n);
  if (err != GA_NO_ERROR) goto error;

  for (i = 0; i < nd; i++) {
    err = GpuKernel_setarg(k, p++, &dims[i]);
    if (err != GA_NO_ERROR) goto error;
  }

  for (j = 0; j < ge->n; j++) {
    if (is_array(ge->args[j])) {
      GpuArray *v = (GpuArray *)args[j];
      err = GpuKernel_setarg(k, p++, v->data);
      if (err != GA_NO_ERROR) goto error;
      err = GpuKernel_setarg(k, p++, &v->offset);
      if (err != GA_NO_ERROR) goto error;
      for (i = 0; i < nd; i++) {
        err = GpuKernel_setarg(k, p++, &strs[j][i]);
        if (err != GA_NO_ERROR) goto error;
      }
    } else {
      err = GpuKernel_setarg(k, p++, args[j]);
      if (err != GA_NO_ERROR) goto error;
    }
  }

  err = GpuKernel_sched(k, n, &ls, &gs);
  if (err != GA_NO_ERROR) goto error;

  err = GpuKernel_call(k, 1, &ls, &gs, 0, NULL);
 error:
  return err;
}

static int gen_elemwise_contig_kernel(GpuKernel *k,
                                      gpucontext *ctx, char **err_str,
                                      const char *preamble,
                                      const char *expr,
                                      unsigned int n,
                                      gpuelemwise_arg *args,
                                      int gen_flags) {
  strb sb = STRB_STATIC_INIT;
  int *ktypes = NULL;
  unsigned int p;
  unsigned int j;
  int flags = GA_USE_CLUDA;
  int res = GA_MEMORY_ERROR;

  flags |= gpuarray_type_flagsa(n, args);

  p = 1;
  for (j = 0; j < n; j++)
    p += ISSET(args[j].flags, GE_SCALAR) ? 1 : 2;

  ktypes = calloc(p, sizeof(int));
  if (ktypes == NULL)
    goto bail;

  p = 0;

  if (preamble)
    strb_appends(&sb, preamble);
  strb_appends(&sb, "\nKERNEL void elem(const ga_size n, ");
  ktypes[p++] = GA_SIZE;
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "GLOBAL_MEM %s *%s_p,  const ga_size %s_offset",
                   ctype(args[j].typecode), args[j].name, args[j].name);
      ktypes[p++] = GA_BUFFER;
      ktypes[p++] = GA_SIZE;
    } else {
      strb_appendf(&sb, "%s %s", ctype(args[j].typecode), args[j].name);
      ktypes[p++] = args[j].typecode;
    }
    if (j != (n - 1))
      strb_appends(&sb, ", ");
  }
  strb_appends(&sb, ") {\n"
               "const ga_size idx = LDIM_0 * GID_0 + LID_0;\n"
               "const ga_size numThreads = LDIM_0 * GDIM_0;\n"
               "ga_size i;\n"
               "GLOBAL_MEM char *tmp;\n\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "tmp = (GLOBAL_MEM char *)%s_p;"
                   "tmp += %s_offset; %s_p = (GLOBAL_MEM %s *)tmp;",
                   args[j].name, args[j].name, args[j].name,
                   ctype(args[j].typecode));
    }
  }

  strb_appends(&sb, "for (i = idx; i < n; i += numThreads) {\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "%s %s;\n", ctype(ISSET(gen_flags, GEN_CONVERT_F16) && args[j].typecode == GA_HALF ?
                                          GA_FLOAT : args[j].typecode), args[j].name);
      if (ISSET(args[j].flags, GE_READ)) {
        if (args[j].typecode == GA_HALF && ISSET(gen_flags, GEN_CONVERT_F16)) {
          strb_appendf(&sb, "%s = load_half(&%s_p[i]);\n", args[j].name, args[j].name);
        } else {
          strb_appendf(&sb, "%s = %s_p[i];\n", args[j].name, args[j].name);
        }
      }
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n");

  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      if (ISSET(args[j].flags, GE_WRITE)) {
        if (args[j].typecode == GA_HALF && ISSET(gen_flags, GEN_CONVERT_F16)) {
          strb_appendf(&sb, "store_half(&%s_p[i], %s);\n", args[j].name, args[j].name);
        } else {
          strb_appendf(&sb, "%s_p[i] = %s;\n", args[j].name, args[j].name);
        }
      }
    }
  }
  strb_appends(&sb, "}\n}\n");

  if (strb_error(&sb))
    goto bail;

  res = GpuKernel_init(k, ctx, 1, (const char **)&sb.s, &sb.l, "elem",
                       p, ktypes, flags, err_str);
 bail:
  strb_clear(&sb);
  free(ktypes);
  return res;
}

static int check_contig(GpuElemwise *ge, void **args,
                        size_t *_n, int *contig) {
  GpuArray *a = NULL, *v;
  size_t n = 1;
  unsigned int i, j;
  int c_contig = 1, f_contig = 1;

  for (i = 0; i < ge->n; i++) {
    if (is_array(ge->args[i])) {
      v = (GpuArray *)args[i];
      if (a == NULL) {
        a = v;
        for (j = 0; j < a->nd; j++) n *= a->dimensions[j];
      }
      c_contig &= GpuArray_IS_C_CONTIGUOUS(v);
      f_contig &= GpuArray_IS_F_CONTIGUOUS(v);
      if (a != v) {
        if (a->nd != v->nd)
          return GA_INVALID_ERROR;
        for (j = 0; j < a->nd; j++) {
          if (v->dimensions[j] != a->dimensions[j])
            return GA_VALUE_ERROR;
        }
      }
    }
  }
  *contig = f_contig || c_contig;
  *_n = n;
  return GA_NO_ERROR;
}

static int call_contig(GpuElemwise *ge, void **args, size_t n) {
  GpuArray *a;
  size_t ls = 0, gs = 0;
  unsigned int i, p;
  int err;

  p = 0;
  err = GpuKernel_setarg(&ge->k_contig, p++, &n);
  if (err != GA_NO_ERROR) return err;
  for (i = 0; i < ge->n; i++) {
    if (is_array(ge->args[i])) {
      a = (GpuArray *)args[i];
      err = GpuKernel_setarg(&ge->k_contig, p++, a->data);
      if (err != GA_NO_ERROR) return err;
      err = GpuKernel_setarg(&ge->k_contig, p++, &a->offset);
      if (err != GA_NO_ERROR) return err;
    } else {
      err = GpuKernel_setarg(&ge->k_contig, p++, args[i]);
      if (err != GA_NO_ERROR) return err;
    }
  }
  err = GpuKernel_sched(&ge->k_contig, n, &ls, &gs);
  if (err != GA_NO_ERROR) return err;
  return GpuKernel_call(&ge->k_contig, 1, &ls, &gs, 0, NULL);
}

GpuElemwise *GpuElemwise_new(gpucontext *ctx,
                             const char *preamble, const char *expr,
                             unsigned int n, gpuelemwise_arg *args,
                             unsigned int nd, int flags) {
  GpuElemwise *res;
#ifdef DEBUG
  char *errstr = NULL;
#endif
  unsigned int i;
  int ret;

  res = calloc(1, sizeof(*res));
  if (res == NULL) return NULL;

  res->flags = flags;
  res->nd = 8;
  res->n = n;

  res->expr = strdup(expr);
  if (res->expr == NULL)
    goto fail;
  if (preamble != NULL) {
    res->preamble = strdup(preamble);
    if (res->preamble == NULL)
      goto fail;
  }

  res->args = copy_args(n, args);
  if (res->args == NULL)
    goto fail;

  /* Count the arrays in the arguements */
  res->narray = 0;
  for (i = 0; i < res->n; i++)
    if (is_array(res->args[i])) res->narray++;

  while (res->nd < nd) res->nd *= 2;
  res->dims = calloc(res->nd, sizeof(size_t));
  if (res->dims == NULL)
    goto fail;
  res->strides = strides_array(res->narray, res->nd);
  if (res->strides == NULL)
    goto fail;
  res->k_basic = calloc(res->nd, sizeof(GpuKernel));
  if (res->k_basic == NULL)
    goto fail;

  res->k_basic_32 = calloc(res->nd, sizeof(GpuKernel));
  if (res->k_basic_32 == NULL)
    goto fail;

  ret = gen_elemwise_contig_kernel(&res->k_contig, ctx,
#ifdef DEBUG
                                   &errstr,
#else
                                   NULL,
#endif
                                   res->preamble, res->expr,
                                   res->n, res->args,
                                   (res->flags & GE_CONVERT_F16));
  if (ret != GA_NO_ERROR) {
#ifdef DEBUG
    if (errstr != NULL)
      fprintf(stderr, "%s\n", errstr);
    free(errstr);
#endif
    goto fail;
  }

  if (ISCLR(flags, GE_NOADDR64)) {
    for (i = 0; i < nd; i++) {
      ret = gen_elemwise_basic_kernel(&res->k_basic[i], ctx,
#ifdef DEBUG
                                      &errstr,
#else
                                      NULL,
#endif
                                      res->preamble, res->expr,
                                      i+1, res->n, res->args,
                                      (res->flags & GE_CONVERT_F16));
      if (ret != GA_NO_ERROR) {
#ifdef DEBUG
        if (errstr != NULL)
          fprintf(stderr, "%s\n", errstr);
        free(errstr);
#endif
        goto fail;
      }
    }
  }

  for (i = 0; i < nd; i++) {
    ret = gen_elemwise_basic_kernel(&res->k_basic_32[i], ctx,
#ifdef DEBUG
                                    &errstr,
#else
                                    NULL,
#endif
                                    res->preamble, res->expr,
                                    i+1, res->n, res->args,
                                    GEN_ADDR32 | (res->flags & GE_CONVERT_F16));
    if (ret != GA_NO_ERROR) {
#ifdef DEBUG
      if (errstr != NULL)
        fprintf(stderr, "%s\n", errstr);
      free(errstr);
#endif
      goto fail;
    }
  }

  return res;

 fail:
  GpuElemwise_free(res);
  return NULL;
}

void GpuElemwise_free(GpuElemwise *ge) {
  unsigned int i;
  for (i = 0; i < ge->nd; i++) {
    if (k_initialized(&ge->k_basic_32[i]))
      GpuKernel_clear(&ge->k_basic_32[i]);
    if (k_initialized(&ge->k_basic[i]))
      GpuKernel_clear(&ge->k_basic[i]);
  }
  if (ge->strides != NULL)
    for (i = 0; i < ge->narray; i++) {
      free(ge->strides[i]);
    }
  if (k_initialized(&ge->k_contig))
    GpuKernel_clear(&ge->k_contig);
  free_args(ge->n, ge->args);
  free((void *)ge->preamble);
  free((void *)ge->expr);
  free(ge->dims);
  free(ge->strides);
  free(ge);
}

int GpuElemwise_call(GpuElemwise *ge, void **args, int flags) {
  size_t n;
  size_t *dims;
  ssize_t **strides;
  unsigned int nd;
  int contig;
  int call32;
  int err;

  err = check_contig(ge, args, &n, &contig);
  if (err == GA_NO_ERROR && contig) {
    if (n == 0) return GA_NO_ERROR;
    return call_contig(ge, args, n);
  }
  err = check_basic(ge, args, flags, &n, &nd, &dims, &strides, &call32);
  if (err == GA_NO_ERROR) {
    if (n == 0) return GA_NO_ERROR;
    return call_basic(ge, args, n, nd, dims, strides, call32);
  }
  return err;
}
