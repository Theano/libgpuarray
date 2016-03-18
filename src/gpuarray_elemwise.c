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

static inline const char *ctype(int typecode) {
  return gpuarray_get_type(typecode).cluda_name;
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
    if (ISCLR(args[j].flags, GE_SCALAR)) {
      strb_appendf(&sb, "tmp = (GLOBAL_MEM char *)%s_data; tmp += %s_offset; "
                   "%s_data = (GLOBAL_MEM *%s)tmp;\n", args[j].name,
                   args[j].name, args[j].name, ctype(args[j].typecode));
    }
  }

  strb_appends(&sb, "for i = idx; i < n; i += numThreads) {\n");
  if (nd > 0)
    strb_appends(&sb, "int ii = i;\nint pos;\n");
  for (j = 0; j < n; j++) {
    if (ISCLR(args[j].flags, GE_SCALAR))
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
      if (ISCLR(args[j].flags, GE_SCALAR))
        strb_appendf(&sb, "%s_p += pos * %s_str_%u;\n", args[j].name,
                     args[j].name, i);
    }
  }
  for (j = 0; j < n; j++) {
    if (ISCLR(args[j].flags, GE_SCALAR)) {
      decl(sb, types[j], isarray[j]);
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
                                      gpuelemwise_args *args) {
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
    if (ISCLR(args[j].flags, GE_SCALAR)) {
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
    if (ISCLR(args[j].flags, GE_SCALAR)) {
      strb_appendf(&sb, "tmp = (GLOBAL_MEM char *)%s_p;"
                   "tmp += %s_offset; %s_p = (GLOBAL_MEM %s*)tmp;",
                   args[j].name, args[j].name, args[j].name,
                   ctype(args[j].typecode));
    }
  }

  strb_appends(&sb, "for (i = idx; i < n; i += numThreads) {\n");
  for (j = 0; j < n; j++) {
    if (ISCLR(args[j].flags, GE_SCALAR)) {
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

int gpuarray_elemwise_op(const gpuarray_buffer_ops *ops, void * ctx,
                         const char *preamble, const char *expr,
                         unsigned int n, const in *types, void **args,
                         int flags, char **err_str) {
  // TODO
}
