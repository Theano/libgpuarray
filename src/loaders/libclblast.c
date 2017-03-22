#include <stdlib.h>

#include "libclblast.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
static const char libname[] = "clblast.dll";
#else /* Unix */
#ifdef __APPLE__
static const char libname[] = "libclblast.dylib";
#else
static const char libname[] = "libclblast.so";
#endif
#endif

#define DEF_PROC(ret, name, args) t##name *name

#include "libclblast.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name, e);   \
  if (name == NULL) {                             \
    return e->code;                               \
  }

static int loaded = 0;

int load_libclblast(error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname, e);
  if (lib == NULL)
    return e->code;

  #include "libclblast.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
