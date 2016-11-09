#include "util/dyn_load.h"

#ifdef __unix__

#include <dlcfn.h>

void *ga_load_library(const char *name) {
  return dlopen(name, RTLD_LAZY|RTLD_LOCAL);
}

void *ga_func_ptr(void *h, const char *name) {
  return dlsym(h, name);
}

#else
/* Should be windows */

void *ga_load_library(const char *name) {
  return LoadLibrary(name);
}

void *ga_func_ptr(void *h, const char *name) {
  return (void *)GetProcAddress(h, name);
}

#endif
