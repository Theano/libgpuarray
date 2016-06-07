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


/**
 * Create a new GpuElemwise.
 *
 * This will allocate and initialized a new GpuElemwise object.  This
 * object can be used to run the specified operation on different sets
 * of arrays.
 *
 * The argument descriptor name the arguments and provide their data
 * types and geometry (arrays or scalars).  They also specify if the
 * arguments are used for reading or writing.  An argument can be used
 * for both.
 *
 * The expression is a C-like string performing an operation with
 * scalar values named according to the argument descriptors.  All of
 * the indexing and selection of the right values is handled by the
 * GpuElemwise code.
 *
 * \param ctx the context in which to run the operations
 * \param preamble code to be inserted before the kernel code
 * \param expr the expression to compute
 * \param n the number of arguments
 * \param args the argument descriptors
 * \param nd the number of dimensions to precompile for
 * \param flags see \ref elem_flags "GpuElemwise flags"
 *
 * \returns a new GpuElemwise object or NULL
 */
GPUARRAY_PUBLIC GpuElemwise *GpuElemwise_new(gpucontext *ctx,
                                             const char *preamble,
                                             const char *expr,
                                             unsigned int n,
                                             gpuelemwise_arg *args,
                                             unsigned int nd,
                                             int flags);

/**
 * \defgroup elem_flags GpuElemwise flags
 * @{
 */

/**
 * Don't precompile kernels for 64-bits addressing.
 */
#define GE_NOADDR64    0x0001

/**
 * Convert float16 inputs to float32 for computation.
 */
#define GE_CONVERT_F16 0x0002

/**
 * @}
 */

/**
 * Free all storage associated with a GpuElemwise.
 *
 * \param ge the GpuElemwise object to free.
 */
GPUARRAY_PUBLIC void GpuElemwise_free(GpuElemwise *ge);

/**
 * Run a GpuElemwise on some inputs.
 *
 * \param ge the GpuElemwise to run
 * \param args pointers to the arguments (must macth what was described by
 *             the argument descriptors)
 * \param flags see \ref elem_call_flags "GpuElemwise call flags"
 */
GPUARRAY_PUBLIC int GpuElemwise_call(GpuElemwise *ge, void **args, int flags);


/**
 * \defgroup elem_call_flags GpuElemwise call flags
 * @{
 */


/**
 * Allow broadcasting of dimensions of size 1.
 */
#define GE_BROADCAST   0x0100

/**
 * Disable dimension collapsing (not recommended).
 */
#define GE_NOCOLLAPSE  0x0200

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
