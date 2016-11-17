#include "dyn_load.h"

#if defined(__unix__) || defined(__APPLE__)

#include <dlfcn.h>
#include <stddef.h>
#include <err.h>

void *ga_load_library(const char *name) {
  void *res = dlopen(name, RTLD_LAZY|RTLD_LOCAL);
#ifdef DEBUG
  if (res == NULL)
    warn("dlopen: %s", name);
#endif
  return res;
}

void *ga_func_ptr(void *h, const char *name) {
  void *res = dlsym(h, name);
#ifdef DEBUG
  if (res == NULL)
    warn("dlsym: %s", name);
#endif
  return res;
}

#else

/* Should be windows */
#include <windows.h>

void *ga_load_library(const char *name) {
  return LoadLibrary(name);
}

void *ga_func_ptr(void *h, const char *name) {
  return (void *)GetProcAddress(h, name);
}

#endif
