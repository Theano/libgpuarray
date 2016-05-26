#include <string.h>

#include "gpuarray/extension.h"

typedef struct _ext {
  const char *name;
  void *val;
} ext;

#ifdef WITH_CUDA
extern void cuda_enter(void);
extern void cuda_exit(void);
extern void *cuda_make_ctx(void);
extern void *cuda_get_stream(void);
extern void *cuda_make_buf(void);
extern void *cuda_get_sz(void);
extern void *cuda_wait(void);
extern void *cuda_record(void);
#endif
#ifdef WITH_OPENCL
extern void *cl_make_ctx(void);
extern void *cl_get_stream(void);
extern void *cl_make_buf(void);
extern void *cl_get_buf(void);
#endif

static ext ext_list[] = {
#ifdef WITH_CUDA
  {"cuda_enter", cuda_enter},
  {"cuda_exit", cuda_exit},
  {"cuda_make_ctx", cuda_make_ctx},
  {"cuda_get_stream", cuda_get_stream},
  {"cuda_make_buf", cuda_make_buf},
  {"cuda_get_sz", cuda_get_sz},
  {"cuda_wait", cuda_wait},
  {"cuda_record", cuda_record},
#endif
#ifdef WITH_OPENCL
  {"cl_make_ctx", cl_make_ctx},
  {"cl_get_stream", cl_get_stream},
  {"cl_make_buf", cl_make_buf},
  {"cl_get_buf", cl_get_buf},
#endif
};

#define N_EXT (sizeof(ext_list)/sizeof(ext_list[0]))

void *gpuarray_get_extension(const char *name) {
  unsigned int i;
  for (i = 0; i < N_EXT; i++) {
    if (strcmp(name, ext_list[i].name) == 0) return ext_list[i].val;
  }
  return NULL;
}  
