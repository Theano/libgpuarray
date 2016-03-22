#include <gpuarray/elemwise.h>

#include "util/strb.h"

struct _GpuElemwise {
  const char *expr;
  const char *preamble;
  gpuelemwise_arg *args;
  GpuKernel k_contig;
  unsigned int n;
  int flags;
};

#define is_array(a) (ISCLR((a).flags, GE_SCALAR))

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode)->cluda_name;
}

/* dst has to be zero-initialized on entry */
static int copy_arg(gpuelemwise_arg *dst, gpuelemwise_arg *src) {
  if (src->dims != NULL) {
    dst->dims = memdup(src->dims, n * sizeof(size_t));
    if (dst->dims == NULL) goto fail_dims;
  }

  if (src->strs != NULL) {
    dst->strs = memdup(src->strs, n * sizeof(ssize_t));
    if (dst->strs == NULL) goto fail_strs;
  }

  dst->name = strdup(src->name);
  if (dst->name == NULL)
    goto fail_name;

  dst->nd = src->nd;
  dst->typecode = src->typecode;
  dst->flags = src->flags;

  return 0;

 fail_name:
  free(dst->strs);
  dst->strs = NULL;
 fail_strs:
  free(dst->dims);
  dst->dims = NULL;
 fail_dims:
  return -1;
}

static void clear_arg(gpuelemwise_arg *a) {
  free(a->dims);
  a->dims = NULL;
  free(a->strs);
  a->strs = NULL;
  free(a->name);
  a->name = NULL;
}

static gpuelemwise_args *copy_args(unsigned int n, gpuelemwise_arg *a) {
  gpuelemwise_args *res = calloc(n, sizeof(gpuelemwise_arg));
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
}

static void free_args(unsigned int n, gpuarray_arg *args) {
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
                                     unsigned int n, gpuarray_arg *args) {
  strb sb = STRB_STATIC_INIT;
  unsigned int i, _i, j;
  int *ktypes;
  size_t p;
  int flags = GA_USE_CLUDA;
  int res;

  flags = gpuarray_type_flagsa(n, args);

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
    strb_appendf(&sb "const ga_size dim%u", i);
    ktypes[p++] = GA_SIZE;
  }
  for (j = 0; j < n; j++) {
    if (ISCRL(args[j].flags, GE_SCALAR)) {
      strb_appendf(&sb "GLOBAL_MEM *%s %s_data, const ga_size %s_offset%s",
                   ctype(args[j].typecode), args[j].name, args[j].name,
                   nd == 0 ? "" : ", ");
      ktypes[p++] = GA_BUFFER;
      ktypes[p++] = GA_SIZE;

      for (i = 0; i < nd; i++) {
        strb_appendf(&sb, "const ga_ssize %s_str_%u%s", args[j].name, i,
                     (i == (nd - 1)) ? "", ", ");
        ktypes[p++] = GA_SSIZE;
      }
    } else {
      strb_appendsf(&sb "%s %s", args[j].name);
      ktypes[p++] = types[j];
    }
    if (j != (n - 1)) strb_appends(&sb, ", ");
  }
  strb_appends(&sb, ") {\n"
               "const ga_size idx = LDIM_0 * GID_0 + LID_0;\n"
               "const ga_size numThreads = LDIM_0 * GDIM_0;\n"
               "ga_size i;\n"
               "GLOBAL_MEM char *tmp;\n\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "tmp = (GLOBAL_MEM char *)%s_data; tmp += %s_offset; "
                   "%s_data = (GLOBAL_MEM *%s)tmp;\n", args[j].name,
                   args[j].name, args[j].name, ctype(args[j].typecode));
    }
  }

  strb_appends(&sb, "for i = idx; i < n; i += numThreads) {\n");
  if (nd > 0)
    strb_appends(&sb, "int ii = i;\nint pos;\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j]))
      strb_appendf(&sb, "GLOBAL_MEM char *%s_p = (GLOBAL_MEM char *)%s_data;\n",
                   args[j].name, args[j].name);
  }
  for (_i = nd; _i > 0; _i--) {
    i = _i - 1;
    if (i > 0)
      strb_appendf(&sb, "pos = ii % dim%u;\nii = ii / dim%u;\n", i, i);
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
      strb_appendf(&sb, "GLOBAL_MEM *%s %s = (GLOBAL_MEM *%s)%s_p;\n",
                   ctype(args[j].typecode), args[j].name,
                   ctype(args[j].typecode), args[j].name);
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n}\n}\n");
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

  flags = gpuarray_type_flagsa(n, args);

  p = 1;
  for (j = 0; j < n; j++)
    p += ISSET(args[j].flags & GE_SCALAR) ? 1 : 2;

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
      strb_appendf(&sb, "GLOBAL MEM *%s %s_p,  const ga_size %s_offset",
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
                   "tmp += %s_offset; %s_p = (GLOBAL_MEM %s*)tmp;",
                   args[j].name, args[j].name, args[j].name,
                   ctype(args[j].typecode));
    }
  }

  strb_appends(&sb, "for (i = idx; i < n; i += numThreads) {\n");
  for (j = 0; j < n; j++) {
    if (is_array(args[j])) {
      strb_appendf(&sb, "GLOBAL_MEM *%s %s = &%s_p[i];",
                   ctype(args[j].typecode), args[j].name, args[j].name);
    }
  }
  strb_appends(&sb, expr);
  strb_appends(&sb, ";\n}\n}\n");

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
    if (is_array(a[i])) {
      v = (GpuArray *)args[i];
      if (a != NULL) {
        a = v;
        for (j = 0; j < a->nd; j++) n *= a->dims[j];
      }
      c_contig &= GpuArray_IS_C_CONTIGUOUS(v);
      f_contig &= GpuArray_IS_F_CONTIGUOUS(v);
      if (a != v) {
        if (a->nd != v->nd)
          return GA_INVALID_ERROR;
        for (j = 0; j < a->nd; j++) {
          if (v->dims[j] != a->dims[j])
            return GA_VALUE_ERROR;
        }
      }
    }
  }
  *contig = f_config || c_contig;
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
  err = GpuKernel_call(&ge->k_contig, 1, &ls, &gs, 0, NULL);
  if (err != GA_NO_ERROR) return err;
}

GpuElemwise *GpuElemwise_new(const gpuarray_buffer_ops *ops, void * ctx,
                             const char *preamble, const char *expr,
                             unsigned int n, gpuelemwise_arg *args,
                             int flags) {
  GpuElemwise *res;
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

  ret = gen_elemwise_contig_kernel(&res->k_contig, ops, ctx, NULL,
                                   res->preamble, res->expr,
                                   res->n, res->args);
  if (ret != GA_NO_ERROR)
    goto fail_contig;

  return res;

 fail_contig:
  free_args(res->n, res->args);
 fail_args:
  free(res->preamble);
 fail_preamble:
  free(res->expr);
 fail_expr:
  free(res);
  return NULL;
}

void GpuElemwise_free(GpuElemwise *ge) {
  GpuKernel_clear(&ge->k_contig);
  free_args(ge->n, ge->args);
  free(ge->preamble);
  free(ge->expr);
  free(ge);
}

int GpuElemwise_call(GpuElemwise *ge, void **args, int flags) {
  size_t n;
  int contig;
  it err;
  err = check_contig(ge->args, args, &n, &contig);
  if (err == GA_NO_ERROR && contig) {
    return call_contig(ge, args, n);
  }
  return GA_UNSUPPORTED_ERROR;
  /* WIP
  err = check_basic(ge->args, args, &n, ...);
  if (err == GA_NO_ERROR)
    return call_basic(ge, args, ...);
  */
}
