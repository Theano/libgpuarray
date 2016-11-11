#include <stdlib.h>

#include "libnccl.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static const char libname[] = "nccl.dll";
#else /* Unix */
static const char libname[] = "libnccl.so";
#endif

#define DEF_PROC(ret, name, args) t##name *name

#include "libnccl.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args)            \
  name = (t##name *)ga_func_ptr(lib, #name); \
  if (name == NULL) {                        \
    return GA_LOAD_ERROR;                    \
  }

static int loaded = 0;

int load_libnccl(void) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname);
  if (lib == NULL)
    return GA_LOAD_ERROR;

  #include "libnccl.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
