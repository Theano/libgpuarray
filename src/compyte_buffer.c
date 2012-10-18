#include <string.h>
#include <errno.h>

#include "compyte/buffer.h"
#include "compyte/error.h"

const char *Gpu_error(compyte_buffer_ops *o, gpudata *b, int err) {
  if (err == GA_IMPL_ERROR)
    return o->buffer_error(b);
  else
    return compyte_error_str(err);
}

#ifdef WITH_CUDA
extern compyte_buffer_ops cuda_ops;
#endif
#ifdef WITH_OPENCL
extern compyte_buffer_ops opencl_ops;
#endif

compyte_buffer_ops *compyte_get_ops(const char *name) {
#ifdef WITH_CUDA
  if (strcmp("cuda", name) == 0) return &cuda_ops;
#endif
#ifdef WITH_OPENCL
  if (strcmp("opencl", name) == 0) return &opencl_ops;
#endif
  return NULL;
}
