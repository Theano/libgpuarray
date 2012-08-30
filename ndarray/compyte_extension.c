#include <string.h>

#include "compyte_extension.h"

typedef struct _ext {
  const char *name;
  void *val;
} ext;

#ifdef WITH_CUDA
extern void *cuda_make_buf(void);
extern void *cuda_get_ptr(void);
extern void *cuda_get_sz(void);
extern void *cuda_set_compiler(void);
#endif
#ifdef WITH_OPENCL
extern void *cl_make_buf(void);
extern void *cl_get_buf(void);
#endif

static ext ext_list[] = {
#ifdef WITH_CUDA
  {"cuda_make_buf", cuda_make_buf},
  {"cuda_get_ptr", cuda_get_ptr},
  {"cuda_get_sz", cuda_get_sz},
  {"cuda_set_compiler", cuda_set_compiler},
#endif
#ifdef WITH_OPENCL
  {"cl_make_buf", cl_make_buf},
  {"cl_get_buf", cl_get_buf}
#endif
};

#define N_EXT (sizeof(ext_list)/sizeof(ext_list[0]))

void *compyte_get_extension(const char *name) {
  unsigned int i;
  for (i = 0; i < N_EXT; i++) {
    if (strcmp(name, ext_list[i].name) == 0) return ext_list[i].val;
  }
  return NULL;
}  
