#ifndef UTIL_ERROR_H
#define UTIL_ERROR_H

/* 1024 - 4 for the int that goes after */
#define ERROR_MSGBUF_LEN 1020

typedef struct _error {
  char msg[MSGBUF_LEN];
  int code;
} error;

int error_alloc(error **ctx);
void error_free(error *ctx);
int error_set(error *ctx, int code, const char *msg);
int error_fmt(error *ctx, int code, const char *fmt, ...);

extern error *global_ctx;

#endif
