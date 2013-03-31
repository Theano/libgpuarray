#ifndef COMPYTE_EXTENSIONS_H
#define COMPYTE_EXTENSIONS_H
/** \file extension.h
 *  \brief Extensions access.
 */

#include <compyte/compat.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

/**
 * Obtain a function pointer for an extension.
 *
 * \returns A function pointer or NULL if the extension was not found.
 */
COMPYTE_PUBLIC void * compyte_get_extension(const char *name);

#ifdef __cplusplus
}
#endif

#endif
