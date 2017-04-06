#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <stdarg.h>
#ifdef _MSC_VER
#include <io.h>
#define read _read
#define write _write
#else
#include <unistd.h>
#endif

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

void strb_read(strb *sb, int fd, size_t sz) {
  ssize_t res;
  char *b;
  if (strb_ensure(sb, sz)) return;
  b = sb->s + sb->l;
  sb->l += sz;
  while (sz) {
    res = read(fd, b, sz);
    if (res == -1 || res == 0) {
      if (res == -1 && (errno == EAGAIN || errno == EINTR))
        continue;
      strb_seterror(sb);
      return;
    }
    sz -= (size_t)res;
    b += (size_t)res;
  }
}

int strb_write(int fd, strb *sb) {
  ssize_t res;
  size_t l = sb->l;
  char *b = sb->s;
  while (l) {
    res = write(fd, b, l);
    if (res == -1) {
      if (errno == EAGAIN || errno == EINTR)
        continue;
      return -1;
    }
    l -= (size_t)res;
    b += (size_t)res;
  }
  return 0;
}
