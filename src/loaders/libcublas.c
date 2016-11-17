#include <stdlib.h>

#include "libcublas.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#define DEF_PROC(name, args) t##name *name
#define DEF_PROC_V2(name, args) DEF_PROC(name, args)
#define DEF_PROC_OPT(name, args) DEF_PROC(name, args)

#include "libcublas.fn"

#undef DEF_PROC_OPT
#undef DEF_PROC_V2
#undef DEF_PROC

#define STRINGIFY(X) #X

#define DEF_PROC(name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name); \
  if (name == NULL) {                        \
    return GA_LOAD_ERROR;                    \
  }

#define DEF_PROC_OPT(name, args)                \
  name = (t##name *)ga_func_ptr(lib, #name);

#define DEF_PROC_V2(name, args)                             \
  name = (t##name *)ga_func_ptr(lib, STRINGIFY(name##_v2)); \
  if (name == NULL) {                                       \
    return GA_LOAD_ERROR;                                   \
  }

static int loaded = 0;

int load_libcublas(int major, int minor) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  {
    static const char DIGITS[] = "0123456789";
    char libname[] = "cublas64_??.dll";

    libname[9] = DIGITS[major];
    libname[10] = DIGITS[minor];

    lib = ga_load_library(libname);
  }
#else /* Unix */
#ifdef __APPLE__
  {
    static const char DIGITS[] = "0123456789";
    char libname[] = "/Developer/NVIDIA/CUDA-?.?/lib/libcublas.dylib";
    libname[23] = DIGITS[major];
    libname[25] = DIGITS[minor];
    lib = ga_load_library(libname);
  }
#else
  lib = ga_load_library("libcublas.so");
#endif
#endif
  if (lib == NULL)
    return GA_LOAD_ERROR;

#include "libcublas.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
