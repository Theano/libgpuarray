#ifndef UTIL_ERROR_H
#define UTIL_ERROR_H

#include <errno.h>
#include <string.h>

#include <gpuarray/error.h>

/* 1024 - 4 for the int that goes after */
#define ERROR_MSGBUF_LEN 1020

typedef struct _error {
  char msg[ERROR_MSGBUF_LEN];
  int code;
} error;

int error_alloc(error **e);
void error_free(error *e);
int error_set(error *e, int code, const char *msg);
int error_fmt(error *e, int code, const char *fmt, ...);
int error_vfmt(error *e, int code, const char *fmt, va_list ap);

extern error *global_err;

static inline int error_sys(error *e, const char *msg) {
  return error_fmt(e, GA_SYS_ERROR, "%s: %s", msg, strerror(errno));
}

#endif
