#include "dyn_load.h"

#if defined(__unix__) || defined(__APPLE__)

#include <dlfcn.h>
#include <err.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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

float ga_lib_version(void *h, void *sym) {
  Dl_info dli;
  char *real_path;
  char *dot1;
  char *dot2;
  char *end;
  float res;

  if (!dladdr(sym, &dli))
    return -1;

  real_path = realpath(dli.dli_fname, NULL);
  if (real_path == NULL)
    return -1;

  dot1 = strrchr(real_path, '.');
  if (dot1 == NULL) {
    free(real_path);
    return -1;
  }
  dot1[0] = '\0';

  dot2 = strrchr(real_path, '.');
  if (dot2 == NULL) {
    free(real_path);
    return -1;
  }
  dot1[0] = '.';

  res = strtof(dot2+1, &end);
  if (*end != '\0') {
    free(real_path);
    return -1;
  }

  free(real_path);
  return res;
}

#else

/* Should be windows */
#include <windows.h>
#pragma comment(lib,"Version.lib")

void *ga_load_library(const char *name) {
  return LoadLibrary(name);
}

void *ga_func_ptr(void *h, const char *name) {
  return (void *)GetProcAddress(h, name);
}

float ga_lib_version(void *h, void *sym) {
  char fname[1024];
  char *vinfo;
  size_t vsize;
  VS_FIXEDFILEINFO *vp;
  unsigned int ui;
  float res;

  if (GetModuleFileName(h, fname, sizeof(fname)) == sizeof(fname))
    return -1;

  vsize = GetFileVersionInfoSize(fname, NULL);
  if (vsize == 0)
    return -1;

  vinfo = malloc(vsize);
  if (vinfo == NULL)
    return -1;

  if (!GetFileVersionInfo(fname, 0, vsize, vinfo)) {
    free(vinfo);
    return -1;
  }

  if (!VerQueryValue(vinfo, "\\", &vp, &ui)) {
    free(vinfo);
    return -1;
  }

  res = ( ((HIWORD(vp->dwFileVersionLS) - 10) * 10000) + LOWORD(vp->dwFileVersionLS) ) / 100.0;

  free(vinfo);
  return res;
}

#endif
