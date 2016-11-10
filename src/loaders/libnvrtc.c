#include <stdlib.h>

#include "libcuda.h"
#include "libnvrtc.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

/* This code is strongly inspired from the dynamic loading code in the
 * samples */

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
    char libname[] = "nvrtc64_??.dll";

    libname[8] = DIGITS[major];
    libname[9] = DIGITS[minor];

    lib = ga_load_library(libname);
  }
#else /* Unix */
  lib = ga_load_library("libnvrtc.so");
#endif
  if (lib == NULL)
    return GA_LOAD_ERROR;

  #include "libnvrtc.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
