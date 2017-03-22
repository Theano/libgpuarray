#include <stdlib.h>

#include "libnccl.h"
#include "dyn_load.h"
#include "gpuarray/error.h"

#define DEF_PROC(ret, name, args) t##name *name

#include "libnccl.fn"

#undef DEF_PROC

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(__APPLE__)
/* As far as we know, nccl is not available or buildable on platforms
   other than linux */
int load_libnccl(error *e) {
  return error_set(e, GA_UNSUPPORTED_ERROR,
                   "NCCL is not available on plaforms other than linux.");
}
#else /* Unix */
static const char libname[] = "libnccl.so";

#define DEF_PROC(ret, name, args)                 \
  name = (t##name *)ga_func_ptr(lib, #name, e);   \
  if (name == NULL) {                             \
    return e->code;                               \
  }

static int loaded = 0;

int load_libnccl(error *e) {
  void *lib;

  if (loaded)
    return GA_NO_ERROR;

  lib = ga_load_library(libname, e);
  if (lib == NULL)
    return e->code;

  #include "libnccl.fn"

  loaded = 1;
  return GA_NO_ERROR;
}
#endif
