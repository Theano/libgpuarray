#include <stdio.h>
#include <stdlib.h>

#include "libcuda.h"
#include "dyn_load.h"
#include "gpuarray/error.h"
#include "util/error.h"

/* This code is inspired from the dynamic loading code in the samples */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static char libname[] = "nvcuda.dll";
#else /* Unix */
#ifdef __APPLE__
static char libname[] = "/Library/Frameworks/CUDA.framework/CUDA";
#else
static char libname[] = "libcuda.so";
#endif
#endif

#define DEF_PROC(name, args) t##name *name
#define DEF_PROC_V2(name, args) DEF_PROC(name, args)

#include "libcuda.fn"

#undef DEF_PROC_V2
#undef DEF_PROC

#define STRINGIFY(X) #X

#define DEF_PROC(name, args)                    \
  name = (t##name *)ga_func_ptr(lib, #name, e); \
  if (name == NULL) {                           \
    return e->code;                             \
  }

#define DEF_PROC_V2(name, args)                                \
  name = (t##name *)ga_func_ptr(lib, STRINGIFY(name##_v2), e); \
  if (name == NULL) {                                          \
    return e->code;                                            \
  }

static int loaded = 0;

int load_libcuda(error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname, e);
  if (lib == NULL)
    return e->code;

  #include "libcuda.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
