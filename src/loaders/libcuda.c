#include <stdio.h>
#include <stdlib.h>

#include "libcuda.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

/* This code is inspired from the dynamic loading code in the samples */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static char libname[] = "nvcuda.dll";
#else /* Unix */
#ifdef __APPLE__
static char libname[] = "CUDA.framework/CUDA";
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

#define DEF_PROC(name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name); \
  if (name == NULL) {                        \
    return GA_LOAD_ERROR;                    \
  }

#define DEF_PROC_V2(name, args)                             \
  name = (t##name *)ga_func_ptr(lib, STRINGIFY(name##_v2)); \
  if (name == NULL) {                                       \
    return GA_LOAD_ERROR;                                   \
  }

static int loaded = 0;

int load_libcuda(void) {
  void *lib;
  float v;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname);
  if (lib == NULL)
    return GA_LOAD_ERROR;

  #include "libcuda.fn"

  v = ga_lib_version(lib, cuInit);
  if (v == -1)
    fprintf(stderr, "WARNING: could not determine cuda driver version.  Some versions return bad results, make sure your version is fine\n");
  #ifdef DEBUG
  fprintf(stderr, "CUDA driver version detected: %.2f\n", v);
  #endif

  if (v > 373.06) {
    if (getenv("GPUARRAY_FORCE_CUDA_DRIVER_LOAD") != NULL) {
      fprintf(stderr, "WARNING: loading blacklisted driver because the load was forced.\n");
    } else {
      fprintf(stderr, "ERROR: refusing to load cuda driver library "
              "because the version is blacklisted.  "
              "Versions 373.06 and below are known to be ok.\n"
              "If you want to bypass this check and force the driver load "
              "define GPUARRAY_FORCE_CUDA_DRIVER_LOAD in your environement.\n");
      return GA_LOAD_ERROR;
    }
  }

  loaded = 1;
  return GA_NO_ERROR;
}
