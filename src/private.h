#ifndef _PRIVATE
#define _PRIVATE

/** \cond INTERNAL_DOCS */

/*
 * This file contains function definition that are shared in multiple
 * files but not exposed in the interface.
 */

#include "private_config.h"

#include "gpuarray/array.h"
#include "gpuarray/types.h"
#include "util/strb.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#define ADDR32_MAX   4294967295
#define SADDR32_MIN -2147483648
#define SADDR32_MAX  2147483647

#define STATIC_ASSERT(COND, MSG) typedef char static_assertion_##MSG[2*(!!(COND))-1]

static inline void *memdup(const void *p, size_t s) {
  void *res = malloc(s);
  if (res != NULL)
    memcpy(res, p, s);
  return res;
}

GPUARRAY_LOCAL int GpuArray_is_c_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_f_contiguous(const GpuArray *a);
GPUARRAY_LOCAL int GpuArray_is_aligned(const GpuArray *a);

GPUARRAY_LOCAL extern const gpuarray_type scalar_types[];
GPUARRAY_LOCAL extern const gpuarray_type vector_types[];

/*
 * This function generates the kernel code to perform indexing on var id
 * from planar index 'i' using the dimensions and strides provided.
 */
GPUARRAY_LOCAL void gpuarray_elem_perdim(strb *sb, unsigned int nd,
                                         const size_t *dims,
                                         const ssize_t *str,
                                         const char *id);

GPUARRAY_LOCAL void gpukernel_source_with_line_numbers(unsigned int count,
                                                       const char **news,
                                                       size_t *newl,
                                                       strb *src);

#define ISSET(v, fl) ((v) & (fl))
#define ISCLR(v, fl) (!((v) & (fl)))

#define FLSET(v, fl) (v |= (fl))
#define FLCLR(v, fl) (v &= ~(fl))

#ifdef __cplusplus
}
#endif

/** \endcond */

#endif
