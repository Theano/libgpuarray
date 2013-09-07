#ifndef _BUF
#define _BUF

#include "private.h"

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

typedef struct _buf {
  char *s; /* memory pointer */
  size_t i; /* used size */
  size_t a; /* allocated size */
} buf;

#define BUF_INIT {0}

COMPYTE_LOCAL int buf_alloc(buf *b, size_t sz);
COMPYTE_LOCAL int buf_ensurefree(buf *b, size_t sz);
COMPYTE_LOCAL void buf_free(buf *b);
COMPYTE_LOCAL void buf_clear(buf *b);
COMPYTE_LOCAL int buf_append(buf *r, const buf *b);
COMPYTE_LOCAL int buf_appends(buf *r, const char *s);
COMPYTE_LOCAL int buf_appendb(buf *r, const void *b, size_t sz);
#define buf_append0(b) buf_appendb(b, "", 1);
COMPYTE_LOCAL int buf_appendc(buf *r, char c);

#ifdef __cplusplus
}
#endif

#endif
