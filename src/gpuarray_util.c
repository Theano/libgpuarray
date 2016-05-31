#include <assert.h>

#include "private.h"
#include "util/strb.h"

#include "gpuarray/util.h"
#include "gpuarray/error.h"
#include "gpuarray/kernel.h"
#include "gpuarray/elemwise.h"

/*
 * API version is negative since we are still in the development
 * phase. Once we go stable, this will move to 0 and go up from
 * there.
 */
const int gpuarray_api_major = -9997;
const int gpuarray_api_minor = 1;

static gpuarray_type **custom_types = NULL;
static int n_types = 0;
static gpuarray_type no_type = {NULL, 0, 0, -1};
typedef struct _buf_st { char c; GpuArray *a; } buf_st;
#define BUF_ALIGN (sizeof(buf_st) - sizeof(GpuArray *))
static gpuarray_type buffer_type = {NULL, sizeof(GpuArray *),
                                   BUF_ALIGN, GA_BUFFER};

int gpuarray_register_type(gpuarray_type *t, int *ret) {
  gpuarray_type **tmp;
  tmp = realloc(custom_types, (n_types+1)*sizeof(*tmp));
  if (tmp == NULL) {
    if (ret) *ret = GA_SYS_ERROR;
    return -1;
  }
  custom_types = tmp;
  t->typecode = 512 + n_types;
  custom_types[n_types++] = t;
  return t->typecode;
}

const gpuarray_type *gpuarray_get_type(int typecode) {
  if (typecode <= GA_DELIM) {
    if (typecode == GA_BUFFER)
      return &buffer_type;
    if (typecode < GA_NBASE)
      return &scalar_types[typecode];
    else
      return &no_type;
  } else if (typecode < GA_ENDVEC) {
    if (typecode < GA_NVEC)
      return &vector_types[typecode - 256];
    else
      return &no_type;
  } else {
    if ((typecode - 512) < n_types)
      return custom_types[typecode - 512];
    else
      return &no_type;
  }
}

size_t gpuarray_get_elsize(int typecode) {
  return gpuarray_get_type(typecode)->size;
}

static inline ssize_t ssabs(ssize_t v) {
  return (v < 0 ? -v : v);
}

void gpuarray_elem_perdim(strb *sb, unsigned int nd,
                          const size_t *dims, const ssize_t *str,
                          const char *id) {
  int i;

  if (nd > 0) {
    strb_appendf(sb, "int %si = i;", id);

    for (i = nd-1; i > 0; i--) {
      strb_appendf(sb, "%s %c= ((%si %% %" SPREFIX "u) * "
                   "%" SPREFIX "d);%si = %si / %" SPREFIX "u;", id,
                   (str[i] < 0 ? '-' : '+'), id, dims[i],
                   ssabs(str[i]), id, id, dims[i]);
    }
    strb_appendf(sb, "%s %c= (%si * %" SPREFIX "d);", id,
                 (str[0] < 0 ? '-' : '+'), id, ssabs(str[0]));
  }
}

void gpukernel_source_with_line_numbers(unsigned int count,
                                        const char **news, size_t *newl,
                                        strb *src) {
  unsigned int section, line, i, j;
  size_t len;

  line = 1;  // start the line counter at 1
  for (section = 0; section < count; section++) {
    len = (newl == NULL) ? 0 : newl[section];
    if (len <= 0)
      len = strlen(news[section]);

    i = 0; // position of line-starts within news[section]
    while (i < len) {
      strb_appendf(src, "%04d\t", line);

      for (j = i; j < len && news[section][j] != '\n'; j++);
      strb_appendn(src, news[section]+i, (j-i));
      strb_appendc(src, '\n');

      i = j+1;  // Character after the newline
      line++;
    }
  }
}

static int get_type_flags(int typecode) {
  int flags = 0;
  if (typecode == GA_DOUBLE || typecode == GA_CDOUBLE)
    flags |= GA_USE_DOUBLE;
  if (typecode == GA_HALF)
    flags |= GA_USE_HALF;
  if (typecode == GA_CFLOAT || typecode == GA_CDOUBLE)
    flags |= GA_USE_COMPLEX;
  if (gpuarray_get_elsize(typecode) < 4)
    flags |= GA_USE_SMALL;
  return flags;
}

/* List of typecodes terminated by -1 */
int gpuarray_type_flags(int init, ...) {
  va_list ap;
  int typecode = init;
  int flags = 0;

  va_start(ap, init);
  while (typecode != -1) {
    flags |= get_type_flags(typecode);
    typecode = va_arg(ap, int);
  }
  va_end(ap);
  return flags;
}

int gpuarray_type_flagsa(unsigned int n, gpuelemwise_arg *args) {
  unsigned int i;
  int flags = 0;
  for (i = 0; i < n; i++) {
    flags |= get_type_flags(args[i].typecode);
  }
  return flags;
}

static inline void shiftdown(ssize_t *base, unsigned int i, unsigned int nd) {
  if (base != NULL)
    memmove(&base[i], &base[i+1], (nd - i - 1)*sizeof(size_t));
}

void gpuarray_elemwise_collapse(unsigned int n, unsigned int *_nd,
                                size_t *dims, ssize_t **strs) {
  unsigned int i;
  unsigned int k;
  unsigned int nd = *_nd;

  /* Remove dimensions of size 1 */
  for (i = nd; i > 0; i--) {
    if (nd > 1 && dims[i-1] == 1) {
      shiftdown((ssize_t *)dims, i-1, nd);
      for (k = 0; k < n; k++)
        shiftdown(strs[k], i-1, nd);
      nd--;
    }
  }

  for (i = nd - 1; i > 0; i--) {
    int collapse = 1;
    for (k = 0; k < n; k++) {
      collapse &= (strs[k] == NULL ||
                   strs[k][i - 1] == dims[i] * strs[k][i]);
    }
    if (collapse) {
      dims[i-1] *= dims[i];
      shiftdown((ssize_t *)dims, i, nd);
      for (k = 0; k < n; k++) {
        if (strs[k] != NULL) {
          strs[k][i-1] = strs[k][i];
          shiftdown(strs[k], i, nd);
        }
      }
      nd--;
    }
  }
  *_nd = nd;
}
