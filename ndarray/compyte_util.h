#ifndef COMPYTE_UTIL
#define COMPYTE_UTIL

#include <sys/types.h>

#include "compyte_types.h"

compyte_type *compyte_get_type(int typecode);
size_t compyte_get_elsize(int typecode);

#endif /* COMPYTE_UTIL */
