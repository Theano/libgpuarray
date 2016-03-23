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
 * Elementwise genearator structure.
 *
 * The contents are private.
 */
typedef struct _GpuElemwise GpuElemwise;

/**
 * Argument information structure for GpuElemwise.
 */
typedef struct _gpuelemwise_arg {
  /**
   * Dimension sizes of argument, optional (can be NULL).
   */
  size_t *dims;

  /**
   * Strides of argument, optional (can be NULL).
   */
  ssize_t *strs;

  /**
   * Name of this argument in the associated expression, mandatory.
   */
  const char *name;

  /**
   * Number of dimensions of argument, mandatory.
   */
  unsigned int nd;

  /**
   * Type of argument, mandatory (not GA_BUFFER, the content dtype)
   */
  int typecode;

  /**
   * Padding, do not use (must be 0).
   */
  unsigned int reserved;

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
   * Disable indexing for this argument on dimensions of size 1.
   * Requires dims.
   */
#define GE_NOINDEX1    0x0002

  /**
   * Disable specialization on dimensions for this argument even if
   * dims is provided.
   */
#define GE_NODIMSPEC   0x0004

/**
 * }@
 */

} gpuelemwise_arg;

GPUARRAY_PUBLIC GpuElemwise *GpuElemwise_new(const gpuarray_buffer_ops *ops,
                                             void * ctx,
                                             const char *preamble,
                                             const char *expr,
                                             unsigned int n,
                                             gpuelemwise_arg *args, int flags);

GPUARRAY_PUBLIC void GpuElemwise_free(GpuElemwise *ge);

GPUARRAY_PUBLIC int GpuElemwise_call(GpuElemwise *ge, void **args, int flags);

#define GE_BROADCAST   0x0100
#define GE_NOCOLLAPSE  0x0200

#ifdef __cplusplus
}
#endif

#endif
