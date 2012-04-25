#ifndef COMPYTE_COMPAT
#define COMPYTE_COMPAT

#ifdef _MSC_VER
#include <stddef.h>
#include "wincompat/stdint.h"
#define ssize_t intptr_t
#define SSIZE_MAX INTPTR_MAX
#else
#include <sys/types.h>
#include <stdint.h>
#endif

#ifdef _MSC_VER
int asprintf(char **ret, const char *fmt, ...);
#endif

#include <stdio.h>

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
#ifdef NO_STRL
size_t strlcpy(char *dst, const char *src, size_t size);
size_t strlcat(char *dst, const char *src, size_t size);
#endif

#endif
