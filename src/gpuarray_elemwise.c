#include <gpuarray/elemwise.h>
#include <gpuarray/array.h>
#include <gpuarray/error.h>
#include <gpuarray/kernel.h>
#include <gpuarray/util.h>

#include "private.h"
#include "util/strb.h"

struct _GpuElemwise {
  const char *expr;
  const char *preamble;
  gpuelemwise_arg *args;
  GpuKernel k_contig;
  GpuKernel *k_basic;
  GpuKernel *k_basic_32;
  unsigned int n;
  unsigned int nd;
  int flags;
};

#define GEN_ADDR32 0x1

#define is_array(a) (ISCLR((a).flags, GE_SCALAR))

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

  for (i = 0; i < n; i++)
    clear_arg(&args[i]);
  free(args);
}

static int gen_elemwise_basic_kernel(GpuKernel *k,
                                     const gpuarray_buffer_ops *ops,
                                     void *ctx, char **err_str,
                                     const char *preamble,
                                     const char *expr,
                                     unsigned int nd,
                                     unsigned int n, gpuelemwise_arg *args,
                                     int gen_flags) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *ktypes;
  size_t p;
  char *size = "ga_size", *ssize = "ga_ssize";
  int sz_code = GA_SIZE, ssz_code = GA_SSIZE;
  int flags = GA_USE_CLUDA;
  int res;

  if (ISSET(gen_flags, GEN_ADDR32)) {
    size = "ga_uint";
    ssize = "ga_int";
    sz_code = GA_UINT;
    ssz_code = GA_INT;
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
  strb_appendf(&sb, "\nKERNEL void elem(const %s n, ", size);
  ktypes[p++] = sz_code;
  for (i = 0; i < nd; i++) {
    strb_appendf(&sb, "const %s dim%u, ", size, i);
    ktypes[p++] = sz_code;
  }
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "GLOBAL_MEM %s *%s_data, const %s %s_offset%s",
                   ctype(args[j].typecode), args[j].name, size, args[j].name,
                   nd == 0 ? "" : ", ");
      ktypes[p++] = GA_BUFFER;
      ktypes[p++] = sz_code;

      for (i = 0; i < nd; i++) {
        strb_appendf(&sb, "const %s %s_str_%u%s", ssize, args[j].name, i,
                     (i == (nd - 1)) ? "": ", ");
        ktypes[p++] = ssz_code;
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
      strb_appendf(&sb, "pos = ii %% dim%u;\nii = ii / dim%u;\n", i, i);
    else
      strb_appends(&sb, "pos = ii;\n");
    for (j = 0; j < n; j++) {
      if (is_array(args[j]))
        strb_appendf(&sb, "%s_p += pos * %s_str_%u;\n", args[j].name,
                     args[j].name, i);
    }
  }
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "%s %s;", ctype(args[j].typecode), args[j].name);
      if (ISSET(args[j].flags, GE_READ)) {
        strb_appendf(&sb, "%s = *(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)%s_data) + %s_p);\n",
                     args[j].name, ctype(args[j].typecode), args[j].name, args[j].name);
      }
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j]) && ISSET(args[j].flags, GE_WRITE)) {
      strb_appendf(&sb, "*(GLOBAL_MEM %s *)(((GLOBAL_MEM char *)%s_data) + %s_p) = %s;\n",
                   ctype(args[j].typecode), args[j].name, args[j].name, args[j].name);
    }
  }
  strb_appends(&sb, "}\n}\n");
  if (strb_error(&sb)) {
    res = GA_MEMORY_ERROR;
    goto bail;
  }

  res = GpuKernel_init(k, ops, ctx, 1, (const char **)&sb.s, &sb.l, "elem",
                       p, ktypes, flags, err_str);
 bail:
  free(ktypes);
  strb_clear(&sb);
  return res;
}

static ssize_t **strides_array(unsigned int num, unsigned int nd) {
  ssize_t **res = calloc(num, sizeof(ssize_t *) + 1);
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
  ssize_t **strs;
  size_t *dims;
  size_t n;
  GpuArray *a = NULL, *v;
  unsigned int i, j, p, num_arrays = 0, nd = 0;
  int err;
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

  /* Now we know that there is at least one array argument and that
     all array arguments have the same number of dimensions */

  /* Allocate the dims and strides buffer */
  dims = calloc(nd, sizeof(size_t));
  if (dims == NULL)
    return GA_MEMORY_ERROR;

  strs = strides_array(num_arrays, nd);
  if (strs == NULL) {
    free(dims);
    return GA_MEMORY_ERROR;
  }

  /* And copy their initial values in */
  memcpy(dims, a->dimensions, nd*sizeof(size_t));
  p = 0;
  for (i = 0; i < ge->n; i++) {
    if (is_array(ge->args[i])) {
      memcpy(strs[p], ((GpuArray *)args[i])->strides, nd*sizeof(ssize_t));
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
        if (dims[j] != v->dimensions[j]) {
          if (ISCLR(flags, GE_BROADCAST)) {
            err = GA_VALUE_ERROR;
            goto error;
          }
          /* GE_BROADCAST is set */
          if (dims[j] == 1) {
            dims[j] = v->dimensions[j];
          } else {
            if (v->dimensions[j] != 1) {
              err = GA_VALUE_ERROR;
              goto error;
            }
          }
        }
        /* If the dimension is 1 set the strides to 0 regardless since
           it won't change anything in the non-broadcast case. */
        if (v->dimensions[j] == 1) {
          strs[p][j] = 0;
        }
        call32 &= v->offset < ADDR32_MAX;
        call32 &= (SADDR32_MIN < strs[p][j] && strs[p][j] < SADDR32_MAX);
        p++;
      } /* is_array() */
    } /* for each arg */
    /* We have the final value in dims[j] */
    n *= dims[j];
  } /* for each dim */

  call32 &= n < ADDR32_MAX;

  if (ISCLR(flags, GE_NOCOLLAPSE) && nd > 1) {
    gpuarray_elemwise_collapse(num_arrays, &nd, dims, strs);
  }

  if (call32) {
    /* Convert our data in-place to 32 bits */
    *((unsigned int *)_n) = n;
    for (j = 0; j < nd; j++) {
      *((unsigned int *)&dims[i]) = dims[i];
      for (p = 0; p < num_arrays; p++) {
        *((int *)&strs[p][j]) = strs[p][j];
      }
    }
  } else {
    *_n = n;
  }
  *_nd = nd;
  *_dims = dims;
  *_strides = strs;
  *_call32 = call32;

  return GA_NO_ERROR;
 error:
  free(dims);
  if (strs != NULL)
    for (p = 0; p < num_arrays; p++) free(strs[p]);
  free(strs);
  return err;
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
                                      const gpuarray_buffer_ops *ops,
                                      void *ctx, char **err_str,
                                      const char *preamble,
                                      const char *expr,
                                      unsigned int n,
                                      gpuelemwise_arg *args) {
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
      strb_appendf(&sb, "%s %s;\n", ctype(args[j].typecode), args[j].name);
      if (ISSET(args[j].flags, GE_READ))
        strb_appendf(&sb, "%s = %s_p[i];\n", args[j].name, args[j].name);
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n");

  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      if (ISSET(args[j].flags, GE_WRITE))
        strb_appendf(&sb, "%s_p[i] = %s;\n", args[j].name, args[j].name);
    }
  }
  strb_appends(&sb, "}\n}\n");

  if (strb_error(&sb))
    goto bail;

  res = GpuKernel_init(k, ops, ctx, 1, (const char **)&sb.s, &sb.l, "elem",
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

GpuElemwise *GpuElemwise_new(const gpuarray_buffer_ops *ops, void * ctx,
                             const char *preamble, const char *expr,
                             unsigned int n, gpuelemwise_arg *args,
                             unsigned int nd, int flags) {
  GpuElemwise *res;
  unsigned int i;
  int ret;

  res = malloc(sizeof(*res));
  if (res == NULL) return NULL;

  res->flags = flags;

  res->expr = strdup(expr);
  if (res->expr == NULL)
    goto fail_expr;
  if (preamble == NULL) {
    res->preamble = NULL;
  } else {
    res->preamble = strdup(preamble);
    if (res->preamble == NULL)
      goto fail_preamble;
  }

  res->n = n;
  res->args = copy_args(n, args);
  if (res->args == NULL)
    goto fail_args;

  res->nd = nd;
  res->k_basic = calloc(nd, sizeof(GpuKernel));
  if (res->k_basic == NULL)
    goto fail_basicl;

  res->k_basic_32 = calloc(nd, sizeof(GpuKernel));
  if (res->k_basic_32 == NULL)
    goto fail_basic32l;

  ret = gen_elemwise_contig_kernel(&res->k_contig, ops, ctx, NULL,
                                   res->preamble, res->expr,
                                   res->n, res->args);
  if (ret != GA_NO_ERROR)
    goto fail_contig;

  for (i = 0; i < nd; i++) {
    ret = gen_elemwise_basic_kernel(&res->k_basic[i], ops, ctx, NULL,
                                    res->preamble, res->expr,
                                    i+1, res->n, res->args, 0);
    if (ret != GA_NO_ERROR)
      goto fail_basic_gen;
  }

  for (i = 0; i < nd; i++) {
    ret = gen_elemwise_basic_kernel(&res->k_basic_32[i], ops, ctx, NULL,
                                    res->preamble, res->expr,
                                    i+1, res->n, res->args, GEN_ADDR32);
    if (ret != GA_NO_ERROR)
      goto fail_basic_gen32;
  }

  return res;

fail_basic_gen32:
  for (; i > 0; i--) {
    GpuKernel_clear(&res->k_basic_32[i-1]);
  }
  i = nd;
fail_basic_gen:
  for (; i > 0; i--) {
    GpuKernel_clear(&res->k_basic[i-1]);
  }
  GpuKernel_clear(&res->k_contig);
 fail_contig:
  free(res->k_basic_32);
 fail_basic32l:
  free(res->k_basic);
 fail_basicl:
  free_args(res->n, res->args);
 fail_args:
  free((void *)res->preamble);
 fail_preamble:
  free((void *)res->expr);
 fail_expr:
  free(res);
  return NULL;
}

void GpuElemwise_free(GpuElemwise *ge) {
  unsigned int i;
  for (i = 0; i < ge->nd; i++) {
    GpuKernel_clear(&ge->k_basic_32[i]);
  }
  for (i = 0; i < ge->nd; i++) {
    GpuKernel_clear(&ge->k_basic[i]);
  }
  GpuKernel_clear(&ge->k_contig);
  free_args(ge->n, ge->args);
  free((void *)ge->preamble);
  free((void *)ge->expr);
  free(ge);
}

int GpuElemwise_call(GpuElemwise *ge, void **args, int flags) {
  size_t n;
  size_t *dims;
  ssize_t **strides;
  unsigned int nd, i;
  int contig;
  int call32;
  int err;

  err = check_contig(ge, args, &n, &contig);
  if (err == GA_NO_ERROR && contig) {
    return call_contig(ge, args, n);
  }
  err = check_basic(ge, args, flags, &n, &nd, &dims, &strides, &call32);
  if (err == GA_NO_ERROR) {
    err =  call_basic(ge, args, n, nd, dims, strides, call32);
    free(dims);
    for (i = 0; strides[i] != NULL; i++) free(strides[i]);
    free(strides);
  }
  return err;
}
