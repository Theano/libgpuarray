#include <stdlib.h>

#include "libopencl.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static char libname[] = "OpenCL.dll";
#else /* Unix */
#ifdef __APPLE__
static char libname[] = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
#else
static char libname[] = "libOpenCL.so";
#endif
#endif

#define DEF_PROC(ret, name, args) t##name *name

#include "libopencl.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name, e);   \
  if (name == NULL) {                             \
    return e->code;                               \
  }

static int loaded = 0;

int load_libopencl(error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname, e);
  if (lib == NULL)
    return e->code;

  #include "libopencl.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
