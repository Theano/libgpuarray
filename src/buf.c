#include <stdlib.h>
#include <string.h>

#include "private.h"
#include "buf.h"

#define DOUBLE_NO_OVERFLOW (1UL << ((sizeof(size_t) * 8) - 1))
#define SAFE_DOUBLE(a, ret) if ((a) >= DOUBLE_NO_OVERFLOW) { return ret; } else a *= 2

#define INIT_SIZE 512

int buf_alloc(buf *b, size_t sz) {
  char *tmp;
  size_t req;
  if (b->a < sz) {
    /* When we need to realloc we double the size (at least) to avoid
       doing it a lot */
    req = b->a;
    SAFE_DOUBLE(req, -1);
    if (req == 0)
      req = INIT_SIZE;
    while (req < sz) {
      SAFE_DOUBLE(req, -1);
    }
    tmp = realloc(b->s, req);
    if (tmp == NULL)
      return -1;
    b->s = tmp;
    b->a = req;
  }
  return 0;
}

int buf_ensurefree(buf *b, size_t sz) {
  if ((b->a - b->i) < sz) {
    return buf_alloc(b, b->a+sz);
  }
  return 0;
}

void buf_free(buf *b) {
  free(b->s);
  b->s = NULL;
  b->i = 0;
  b->a = 0;
}

void buf_clear(buf *b) {
  b->i = 0;
}

int buf_append(buf *r, const buf *b) {
  return buf_appendb(r, b->s, b->i);
}

int buf_appends(buf *r, const char *s) {
  return buf_appendb(r, s, strlen(s));
}

int buf_appendb(buf *r, const void *b, size_t sz) {
  if (buf_ensurefree(r, sz))
    return -1;
  memcpy(r->s+r->i, b, sz);
  r->i += sz;
  return 0;
}

int buf_appendc(buf *r, char c) {
  if (buf_ensurefree(r, 1))
    return -1;
  r->s[r->i++] = c;
  return 0;
}
