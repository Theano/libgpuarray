
#include <stdarg.h>
#include "util/strb.h"

strb *strb_alloc(size_t i) {
  strb *res = malloc(sizeof(strb));
  if (res == NULL) return NULL;
  res->s = malloc(i);
  if (res->s == NULL) { free(res); return NULL; }
  res->a = i;
  res->l = 0;
  return res;
}

void strb_free(strb *sb) {
  free(sb->s);
  free(sb);
}

int strb_grow(strb *sb, size_t n) {
  char *s;
  if (strb_error(sb)) return -1;
  if (sb->a == 0 && n < 1024) n = 1024;
  if (sb->a > n) n = sb->a;
  /* overflow */
  if (SIZE_MAX - sb->a < n) { strb_seterror(sb); return -1; }
  s = realloc(sb->s, sb->a + n);
  if (s == NULL) {
    strb_seterror(sb);
    return -1;
  }
  sb->s = s;
  sb->a += n;
  return 0;
}

void strb_appendf(strb *sb, const char *f, ...) {
  va_list ap;
  int s;

  va_start(ap, f);
#ifdef _MSC_VER
  s = _vscprintf(f, ap);
#else
  s = vsnprintf(NULL, 0, f, ap);
#endif
  va_end(ap);

  if (s < 0) { strb_seterror(sb); return; }
  s += 1;
  
  if (strb_ensure(sb, s)) return;
  va_start(ap, f);
  s = vsnprintf(sb->s+sb->l, s, f, ap);
  va_end(ap);
  sb->l += s;
}
