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

int error_fmt(error *e, int code, const char *fmt, ...) {
  va_list ap;

  e->code = code;
  va_start(ap, fmt);
  vsnprintf(e->msg, ERROR_MSGBUF_LEN, fmt, ap);
  va_end(ap);
#ifdef DEBUG
  fprintf(stderr, "ERROR %d: %s\n", e->code, e->msg);
#endif
  return code;
}
