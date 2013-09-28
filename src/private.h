#ifndef _PRIVATE
#define _PRIVATE

/** \cond INTERNAL_DOCS */

/* 
 * This file contains function definition that are shared in multiple
 * files but not exposed in the interface.
 */

#include "compyte/config.h"
#include "compyte/array.h"
#include "compyte/types.h"

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

COMPYTE_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);
COMPYTE_LOCAL int GpuArray_is_aligned(const GpuArray *a);

#ifndef HAVE_ASPRINTF
COMPYTE_LOCAL int asprintf(char **ret, const char *fmt, ...);
#endif

#ifndef HAVE_MKSTEMP
COMPYTE_LOCAL int mkstemp(char *path);
#endif

#ifndef HAVE_STRL
COMPYTE_LOCAL size_t strlcpy(char *dst, const char *src, size_t size);
COMPYTE_LOCAL size_t strlcat(char *dst, const char *src, size_t size);
#endif

COMPYTE_LOCAL extern const compyte_type scalar_types[];
COMPYTE_LOCAL extern const compyte_type vector_types[];

COMPYTE_LOCAL int compyte_elem_perdim(char *strs[], unsigned int *count,
                                      unsigned int nd, const size_t *dims,
                                      const ssize_t *str, const char *id);

#define nelems(a) (sizeof(a)/sizeof(a[0]))

#ifdef __cplusplus
}
#endif

/** \endcond */

#endif
