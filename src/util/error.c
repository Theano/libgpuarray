#include <stdarg.h>
#include <stdlib.h>

#include "error.h"

static error _global_ctx = {};
error *global_ctx = &_global_ctx;

int error_alloc(error **_ctx) {
  error *ctx;
  ctx = calloc(sizeof(error), 1);
  if (ctx == NULL) return -1;
  *_ctx = ctx;
  return 0;
}

void error_free(error *ctx) {
  free(ctx);
}

int error_setall(error *ctx, int code, const char *msg) {
  ctx->code = code;
  strlcpy(ctx->msg, msg, MSGBUF_LEN);
  return code;
}

int error_fmt(error *ctx, int code, const char *fmt, ...) {
  va_arg ap;

  ctx->code = code;
  va_start(ap, fmt);
  vsnprintf(ctx->msg, MSGBUF_LEN, fmt, ap);
  va_end(ap);
  return code;
}
