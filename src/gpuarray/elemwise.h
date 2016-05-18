#ifndef GPUARRAY_ELEMWISE_H
#define GPUARRAY_ELEMWISE_H
/** \file elemwise.h
 *  \brief Custom elementwise operations generator.
 */

#include <gpuarray/buffer.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

struct _GpuElemwise;

/**
 * Elementwise generator structure.
 *
 * The contents are private.
 */
typedef struct _GpuElemwise GpuElemwise;

/**
 * Argument information structure for GpuElemwise.
 */
typedef struct _gpuelemwise_arg {
  /**
   * Name of this argument in the associated expression, mandatory.
   */
  const char *name;

  /**
   * Type of argument, mandatory (not GA_BUFFER, the content dtype)
   */
  int typecode;

  /**
   * Argument flags, mandatory (see \ref eflags).
   */
  int flags;

/**
 * \defgroup eflags GpuElemwise argument flags
 * @{
 */

  /**
   * Argument is a scalar passed from the CPU, requires nd == 0.
   */
#define GE_SCALAR      0x0001

  /**
   * Array is read from in the expression.
   */
#define GE_READ        0x0002

  /**
   * Array is written to in the expression.
   */
#define GE_WRITE       0x0004

/**
 * }@
 */

} gpuelemwise_arg;

GPUARRAY_PUBLIC GpuElemwise *GpuElemwise_new(gpucontext *ctx,
                                             const char *preamble,
                                             const char *expr,
                                             unsigned int n,
                                             gpuelemwise_arg *args,
                                             unsigned int nd,
                                             int flags);

GPUARRAY_PUBLIC void GpuElemwise_free(GpuElemwise *ge);

GPUARRAY_PUBLIC int GpuElemwise_call(GpuElemwise *ge, void **args, int flags);

#define GE_BROADCAST   0x0100
#define GE_NOCOLLAPSE  0x0200

#ifdef __cplusplus
}
#endif

#endif
