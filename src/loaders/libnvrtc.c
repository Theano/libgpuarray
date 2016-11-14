#include <stdlib.h>

#include "libcuda.h"
#include "libnvrtc.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#define DEF_PROC(name, args) t##name *name

#include "libnvrtc.fn"

#undef DEF_PROC

#define DEF_PROC(name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name); \
  if (name == NULL) {                        \
    return GA_LOAD_ERROR;                    \
  }

static int loaded = 0;

int load_libnvrtc(int major, int minor) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  {
    static const char DIGITS[] = "0123456789";
    char libname[] = "nvrtc64_??.dll";

    libname[8] = DIGITS[major];
    libname[9] = DIGITS[minor];

    lib = ga_load_library(libname);
  }
#else /* Unix */
#ifdef __APPLE__
  {
    static const char DIGITS[] = "0123456789";
    /* Try the usual fullpath first */
    char libname[] = "/Developer/NVIDIA/CUDA-?.?/lib/libnvrtc.dylib";
    libname[23] = DIGITS[major];
    libname[25] = DIGITS[minor];
    lib = ga_load_library(libname);
  }
#else
  lib = ga_load_library("libnvrtc.so");
#endif
#endif
  if (lib == NULL)
    return GA_LOAD_ERROR;

  #include "libnvrtc.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
