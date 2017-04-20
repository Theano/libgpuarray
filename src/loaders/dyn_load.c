#include "dyn_load.h"
#include "util/error.h"

#if defined(__unix__) || defined(__APPLE__)

#include <dlfcn.h>
#include <err.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void *ga_load_library(const char *name, error *e) {
  void *res = dlopen(name, RTLD_LAZY|RTLD_LOCAL);
  if (res == NULL)
    error_fmt(e, GA_LOAD_ERROR, "Could not load \"%s\": %s", name, dlerror());
  return res;
}

void *ga_func_ptr(void *h, const char *name, error *e) {
  void *res = dlsym(h, name);
  if (res == NULL)
    error_fmt(e, GA_LOAD_ERROR, "Could not find synbol \"%s\": %s", name, dlerror());
  return res;
}

#else

/* Should be windows */
#include <windows.h>

static inline void error_win(const char* name, error *e) {
  char msgbuf[512];
  DWORD err = GetLastError();
  DWORD len = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|
                             FORMAT_MESSAGE_IGNORE_INSERTS,
                             NULL, err, 0, msgbuf, 512, NULL);
  if (len == 0)
    error_fmt(e, GA_LOAD_ERROR, "Could not load \"%s\": error code %X", name, err);
  else
    error_fmt(e, GA_LOAD_ERROR, "Could not load \"%s\": %s", name, msgbuf);
}

void *ga_load_library(const char *name, error *e) {
  void *res = LoadLibrary(name);
  if (res == NULL)
    error_win(name, e);
  return res;
}

void *ga_func_ptr(void *h, const char *name, error *e) {
  void *res = (void *)GetProcAddress(h, name);
  if (res == NULL)
    error_win(name, e);
  return res;
}

#endif
