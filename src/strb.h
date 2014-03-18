#ifndef STRB_H
#define STRB_H

#include "private.h"

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

COMPYTE_LOCAL strb *strb_alloc(size_t);
COMPYTE_LOCAL void strb_free(strb *);

#define strb_new() strb_alloc(1024)

static inline void strb_reset(strb *sb) {
  sb->l = 0;
}

static inline void strb_clear(strb *sb) {
  free(sb->s);
  sb->s = NULL;
  sb->a = 0;
  sb->l = 0;
}

COMPYTE_LOCAL int strb_grow(strb *, size_t);

static inline int strb_ensure(strb *sb, size_t s) {
  if (sb->a - sb->l < s) return strb_grow(sb, s);
  return 0;
}

static inline int strb_appendc(strb *sb, char c) {
  if (strb_ensure(sb, 1)) return -1;
  sb->s[sb->l++] = c;
  return 0;
}

#define strb_append0(s) strb_appendc(s, '\0')

static inline int strb_appendn(strb *sb, const char *s, size_t n) {
  if (strb_ensure(sb, n)) return -1;
  memcpy(sb->s+sb->l, s, n);
  sb->l += n;
  return 0;
}

static inline int strb_appends(strb *sb, const char *s) {
  return strb_appendn(sb, s, strlen(s));
}

COMPYTE_LOCAL int strb_appendf(strb *, const char *, ...);

static inline const char *strb_cstr(strb *sb) {
  if (strb_append0(sb)) return NULL;
  sb->l--;
  return sb->s;
}

#ifdef __cplusplus
}
#endif

#endif
