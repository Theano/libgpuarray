#ifndef COMPYTE_EXTENSIONS_H
#define COMPYTE_EXTENSIONS_H

#include "compyte_compat.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

COMPYTE_PUBLIC void * compyte_get_extension(const char *name);

#ifdef __cplusplus
}
#endif

#endif
