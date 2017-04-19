#include <stdlib.h>

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
    static const char DIGITS[] = "0123456789";
    char libname[] = "nvrtc64_??.dll";

    libname[8] = DIGITS[major];
    libname[9] = DIGITS[minor];

    lib = ga_load_library(libname, e);
  }
#else /* Unix */
#ifdef __APPLE__
  {
    static const char DIGITS[] = "0123456789";
    /* Try the usual fullpath first */
    char libname[] = "/Developer/NVIDIA/CUDA-?.?/lib/libnvrtc.dylib";
    libname[23] = DIGITS[major];
    libname[25] = DIGITS[minor];
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
