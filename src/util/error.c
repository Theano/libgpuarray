#include <stdarg.h>
#include <stdlib.h>

#include "error.h"

static error _global_err = {};
error *global_err = &_global_err;

int error_alloc(error **_e) {
  error *e;
  e = calloc(sizeof(error), 1);
  if (e == NULL) return -1;
  *_e = e;
  return 0;
}

void error_free(error *e) {
  free(e);
}

int error_setall(error *e, int code, const char *msg) {
  e->code = code;
  strlcpy(e->msg, msg, MSGBUF_LEN);
  return code;
}

int error_fmt(error *e, int code, const char *fmt, ...) {
  va_arg ap;

  e->code = code;
  va_start(ap, fmt);
  vsnprintf(e->msg, MSGBUF_LEN, fmt, ap);
  va_end(ap);
  return code;
}
