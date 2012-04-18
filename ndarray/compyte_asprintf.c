#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "compyte_compat.h"

#include <stdarg.h>
#include <stdlib.h>

int asprintf(char **ret, const char *fmt, ...) {
  va_list ap;
  char buf[1];
  char *res;
  int size;

  va_start(ap, fmt);
  size = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);

  res = malloc(size+1);
  if (res == NULL) return -1;

  va_start(ap, fmt);
  size = vsnprintf(res, size+1, fmt, ap);
  va_end(ap);
  *ret = res;
  return size;
}
