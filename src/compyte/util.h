#ifndef COMPYTE_UTIL
#define COMPYTE_UTIL
/** \file util.h
 *  \brief Utility functions.
 */

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

#include <compyte/compat.h>
#include <compyte/types.h>

/**
 * Registers a type with the kernel machinery.
 *
 * \param t is a preallocated and filled compyte_type structure. The
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
COMPYTE_PUBLIC int compyte_register_type(compyte_type *t, int *ret);
/**
 * Get the type structure for a type.
 *
 * The resulting structure MUST NOT be modified.
 *
 * \param typecode the typecode to get structure for
 *
 * \returns A type structure pointer or NULL
 */
COMPYTE_PUBLIC const compyte_type *compyte_get_type(int typecode);
/**
 * Get the size of one element of a type.
 *
 * The type MUST exist or the program will crash.
 *
 * \param typecode the type to get the element size for
 *
 * \returns the size
 */
COMPYTE_PUBLIC size_t compyte_get_elsize(int typecode);

COMPYTE_LOCAL int compyte_elem_perdim(char *strs[], unsigned int *count,
                                      unsigned int nd, const size_t *dims,
                                      const ssize_t *str, const char *id);

#ifdef __cplusplus
}
#endif

#endif /* COMPYTE_UTIL */
