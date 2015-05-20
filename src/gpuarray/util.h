#ifndef GPUARRAY_UTIL
#define GPUARRAY_UTIL
/** \file util.h
 *  \brief Utility functions.
 */

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#include <gpuarray/config.h>
#include <gpuarray/types.h>

extern GPUARRAY_PUBLIC const int gpuarray_api_major;
extern GPUARRAY_PUBLIC const int gpuarray_api_minor;

/**
 * Registers a type with the kernel machinery.
 *
 * \param t is a preallocated and filled gpuarray_type structure. The
 *   memory can be allocated from static memory as it will never be
 *   freed.
 * \param ret is a pointer where the error code (if any) will be
 *   stored.  It can be NULL in which case no error code will be
 *   returned.  If there is no error then the memory pointed to by
 *   `ret` will be untouched.
 *
 * \returns The type code that corresponds to the registered type.
 * This code is only valid for the duration of the application and
 * cannot be reused between invocation.
 *
 * On error this function will return -1.
 */
GPUARRAY_PUBLIC int gpuarray_register_type(gpuarray_type *t, int *ret);

/**
 * Get the type structure for a type.
 *
 * The resulting structure MUST NOT be modified.
 *
 * \param typecode the typecode to get structure for
 *
 * \returns A type structure pointer or NULL
 */
GPUARRAY_PUBLIC const gpuarray_type *gpuarray_get_type(int typecode);

/**
 * Get the size of one element of a type.
 *
 * If the type does not exists this function returns (size_t)-1.
 *
 * \param typecode the type to get the element size for
 *
 * \returns the size
 */
GPUARRAY_PUBLIC size_t gpuarray_get_elsize(int typecode);

#ifdef __cplusplus
}
#endif

#endif /* GPUARRAY_UTIL */
