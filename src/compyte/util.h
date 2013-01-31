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
 * ## Parameters 
 * - t is a preallocated and filled compyte_type structure. The memory
 *   can be allocated from static memory as it will never be freed.
 * - ret is a pointer where the error code (if any) will be stored.
 *   It can be NULL in which case no error code will be returned.  If
 *   there is no error then the memory pointed to by `ret` will be
 *   untouched.
 *
 * ## Return value
 * Returns the type code that corresponds to the registered type.
 * This code is only valid for the duration of the application and
 * cannot be reused between invocation.
 *
 * ## Errors
 * On error this function will return -1 and 
 */
COMPYTE_PUBLIC int compyte_register_type(compyte_type *t, int *ret);
COMPYTE_PUBLIC compyte_type *compyte_get_type(int typecode);
COMPYTE_PUBLIC size_t compyte_get_elsize(int typecode);

COMPYTE_LOCAL int compyte_elem_perdim(char *strs[], unsigned int *count, unsigned int nd,
			const size_t *dims, const ssize_t *str,
			const char *id);

#ifdef __cplusplus
}
#endif

#endif /* COMPYTE_UTIL */
