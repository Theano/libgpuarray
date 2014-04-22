#ifndef _PRIVATE
#define _PRIVATE

/** \cond INTERNAL_DOCS */

/*
 * This file contains function definition that are shared in multiple
 * files but not exposed in the interface.
 */

#include "gpuarray/config.h"
#include "gpuarray/array.h"
#include "gpuarray/types.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#include "private_config.h"

#ifdef _MSC_VER
struct iovec {
  char *iov_base;
  size_t iov_len;
};
#else
#include <sys/uio.h>
#endif

#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
/* God damn Microsoft ... */
#define snprintf _snprintf
#endif

#ifdef _MSC_VER
/* MS VC++ 2008 does not support inline */
#define inline
#endif

#ifdef _MSC_VER
#define SPREFIX "I"
#else
#define SPREFIX "z"
#endif

#include <string.h>

static inline void *memdup(const void *p, size_t s) {
  void *res = malloc(s);
  if (res != NULL)
    memcpy(res, p, s);
  return res;
}

GPUARRAY_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_aligned(const GpuArray *a);

#ifndef HAVE_ASPRINTF
GPUARRAY_LOCAL int asprintf(char **ret, const char *fmt, ...);
#endif

#ifndef HAVE_MKSTEMP
GPUARRAY_LOCAL int mkstemp(char *path);
#endif

#ifndef HAVE_STRL
GPUARRAY_LOCAL size_t strlcpy(char *dst, const char *src, size_t size);
GPUARRAY_LOCAL size_t strlcat(char *dst, const char *src, size_t size);
#endif

GPUARRAY_LOCAL extern const gpuarray_type scalar_types[];
GPUARRAY_LOCAL extern const gpuarray_type vector_types[];

GPUARRAY_LOCAL int gpuarray_elem_perdim(char *strs[], unsigned int *count,
                                      unsigned int nd, const size_t *dims,
                                      const ssize_t *str, const char *id);

#define nelems(a) (sizeof(a)/sizeof(a[0]))

#ifdef __cplusplus
}
#endif

/** \endcond */

#endif
