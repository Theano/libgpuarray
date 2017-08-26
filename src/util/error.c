#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "private_config.h"
#include "util/error.h"

static error _global_err = {{0}, 0};
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

int error_set(error *e, int code, const char *msg) {
  e->code = code;
  strlcpy(e->msg, msg, ERROR_MSGBUF_LEN);
#ifdef DEBUG
  fprintf(stderr, "ERROR %d: %s\n", e->code, e->msg);
#endif
  return code;
}

int error_vfmt(error *e, int code, const char *fmt, va_list ap) {
  e->code = code;
  vsnprintf(e->msg, ERROR_MSGBUF_LEN, fmt, ap);
#ifdef DEBUG
  fprintf(stderr, "ERROR %d: %s\n", e->code, e->msg);
#endif
  return code;
}

int error_fmt(error *e, int code, const char *fmt, ...) {
  int ret;
  va_list ap;
  va_start(ap, fmt);
  ret = error_vfmt(e, code, fmt, ap);
  va_end(ap);
  return ret;
}
