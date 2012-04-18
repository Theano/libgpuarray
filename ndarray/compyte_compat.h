#ifndef COMPYTE_COMPAT
#define COMPYTE_COMPAT

/* Define size_t and ssize_t */
#ifdef _MSC_VER
#include <stddef.h>
typedef intptr_t ssize_t;
#else
#include <sys/types.h>
#endif

#ifdef __linux__
/* We define _GNU_SOURCE since otherwise stdio.h will not expose
   asprintf on linux. */
#define _GNU_SOURCE
#include <stdio.h>
#undef _GNU_SOURCE
#else
#ifdef _MSC_VER
int asprintf(char **ret, const char *fmt, ...);
#endif
#include <stdio.h>
#endif

#ifdef _MSC_VER
/* MS VC++ 2008 does not support inline */
#define inline 
#endif

#endif
