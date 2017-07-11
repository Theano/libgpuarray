/* To be able to use snprintf with any compiler including MSVC2008. */
#include <private_config.h>

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
    char libname[64];
    int n;
    #ifdef DEBUG
    fprintf(stderr, "Loading nvrtc %d.%d.\n", major, minor);
    #endif
    n = snprintf(libname, sizeof(libname), "nvrtc64_%d%d.dll", major, minor);
    if (n < 0 || n >= sizeof(libname))
      return error_set(e, GA_SYS_ERROR, "snprintf");

    lib = ga_load_library(libname, e);
  }
#else /* Unix */
#ifdef __APPLE__
  {
    char libname[128];
    int n;
    #ifdef DEBUG
    fprintf(stderr, "Loading nvrtc %d.%d.\n", major, minor);
    #endif
    n = snprintf(libname, sizeof(libname), "/Developer/NVIDIA/CUDA-%d.%d/lib/libnvrtc.dylib", major, minor);
    if (n < 0 || n >= sizeof(libname))
      return error_set(e, GA_SYS_ERROR, "snprintf");
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
