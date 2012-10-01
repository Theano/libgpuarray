#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "compyte/compat.h"

#include <stdarg.h>
#include <stdlib.h>

int asprintf(char **ret, const char *fmt, ...) {
  va_list ap;
  char *res;
  int size;

  va_start(ap, fmt);
#ifdef _MSC_VER
  size = _vscprintf(fmt, ap);
#else
  size = vsnprintf(NULL, 0, fmt, ap);
#endif
  va_end(ap);
  if (size < 0) return -1;
  size += 1;

  res = malloc(size);
  if (res == NULL) return -1;

  va_start(ap, fmt);
  size = vsnprintf(res, size, fmt, ap);
  va_end(ap);
  *ret = res;
  return size;
}
