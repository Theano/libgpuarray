#ifndef STRB_H
#define STRB_H

#include "private_config.h"
#include "util/halloc.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

typedef struct _strb {
  char *s;
  size_t l;
  size_t a;
} strb;

#define STRB_STATIC_INIT {NULL, 0, 0}

GPUARRAY_LOCAL strb *strb_alloc(size_t);
GPUARRAY_LOCAL void strb_free(strb *);

#define strb_new() strb_alloc(1024)

static inline void strb_reset(strb *sb) {
  sb->l = 0;
}

static inline int strb_seterror(strb *sb) {
  sb->l = (size_t)-1;
  return -1;
}

static inline int strb_error(strb *sb) {
  return sb->l == (size_t)-1;
}

static inline void strb_clear(strb *sb) {
  h_free(sb->s);
  sb->s = NULL;
  sb->a = 0;
  sb->l = 0;
}

GPUARRAY_LOCAL int strb_grow(strb *, size_t);

static inline int strb_ensure(strb *sb, size_t s) {
  if (strb_error(sb)) return -1;
  if (sb->a - sb->l < s) return strb_grow(sb, s);
  return 0;
}

static inline void strb_appendc(strb *sb, char c) {
  if (strb_ensure(sb, 1)) return;
  sb->s[sb->l++] = c;
}

#define strb_append0(s) strb_appendc(s, '\0')

static inline void strb_appendn(strb *sb, const char *s, size_t n) {
  if (strb_ensure(sb, n)) return;
  memcpy(sb->s+sb->l, s, n);
  sb->l += n;
}

static inline void strb_appends(strb *sb, const char *s) {
  strb_appendn(sb, s, strlen(s));
}

static inline void strb_appendb(strb *sb, strb *sb2) {
  strb_appendn(sb, sb2->s, sb2->l);
}

GPUARRAY_LOCAL void strb_appendf(strb *, const char *, ...);

static inline char *strb_cstr(strb *sb) {
  strb_append0(sb);
  if (strb_error(sb)) {
    strb_clear(sb);
    return NULL;
  }
  sb->l--;
  return sb->s;
}

#ifdef __cplusplus
}
#endif

#endif
