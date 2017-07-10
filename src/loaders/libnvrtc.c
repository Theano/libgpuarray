#include <stdlib.h>
#ifdef DEBUG
/* For fprintf and stderr. */
#include <stdio.h>
#endif

#include "libcuda.h"
#include "libnvrtc.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#define DEF_PROC(rt, name, args) t##name *name

#include "libnvrtc.fn"

#undef DEF_PROC

#define DEF_PROC(rt, name, args)                  \
  name = (t##name *)ga_func_ptr(lib, #name, e);   \
  if (name == NULL) {                             \
    return e->code;                               \
  }

static int loaded = 0;

int load_libnvrtc(int major, int minor, error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  {
    const char* libname_pattern = "nvrtc64_%d%d.dll";
    char libname[64];

    #ifdef DEBUG
    fprintf(stderr, "Loading nvrtc %d.%d.\n", major, minor);
    #endif
    sprintf(libname, libname_pattern, major, minor);

    lib = ga_load_library(libname, e);
  }
#else /* Unix */
#ifdef __APPLE__
  {
    /* Try the usual fullpath first */
    const char* libname_pattern = "/Developer/NVIDIA/CUDA-%d.%d/lib/libnvrtc.dylib";
    char libname[128];
    sprintf(libname, libname_pattern, major, minor);
    lib = ga_load_library(libname, e);
  }
#else
  lib = ga_load_library("libnvrtc.so", e);
#endif
#endif
  if (lib == NULL)
    return e->code;

  #include "libnvrtc.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
