#include <stdlib.h>

#include "libclblas.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static const char libname[] = "clBLAS.dll";
#else /* Unix */
#ifdef __APPLE__
static const char libname[] = "libclBLAS.dylib";
#else
static const char libname[] = "libclBLAS.so";
#endif
#endif

#define DEF_PROC(ret, name, args) t##name *name

#include "libclblas.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name, e);   \
  if (name == NULL) {                             \
    return e->code;                               \
  }

static int loaded = 0;

int load_libclblas(error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname, e);
  if (lib == NULL)
    return e->code;

  #include "libclblas.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
