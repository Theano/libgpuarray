#include <stdlib.h>

#include "libopencl.h"
#include "dyn_load.h"
#include "gpuarray/error.h"
/* This code is strongly inspired from the dynamic loading code in the
 * samples */
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static char libname[] = "OpenCL.dll";
#else /* Unix */
static char libname[] = "libOpenCL.so";
#endif

#define DEF_PROC(ret, name, args) t##name *name

#include "libopencl.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args)            \
  name = (t##name *)ga_func_ptr(lib, #name); \
  if (name == NULL) {                        \
    return GA_LOAD_ERROR;                    \
  }

static int loaded = 0;

int load_libopencl(void) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname);
  if (lib == NULL)
    return GA_LOAD_ERROR;

  #include "libopencl.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
