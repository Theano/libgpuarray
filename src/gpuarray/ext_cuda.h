#ifndef LIBGPU_EXT_CUDA
#define LIBGPU_EXT_CUDA

#include <cuda.h>

#include <gpuarray/config.h>
#include <gpuarray/buffer.h>
#include <gpuarray/extension.h>

#ifdef __cplusplus
extern "C" {
#endif

static void (*cuda_enter)(gpucontext *);
static void (*cuda_exit)(gpucontext *);
static gpucontext *(*cuda_make_ctx)(CUcontext, int);
static CUstream (*cuda_get_stream)(void *);
static gpudata *(*cuda_make_buf)(void *, CUdeviceptr, size_t);
static CUdeviceptr (*cuda_get_ptr)(gpudata *);
static size_t (*cuda_get_sz)(gpudata *);
static int (*cuda_wait)(gpudata *, int);
static int (*cuda_record)(gpudata *, int);

static void setup_ext_cuda(void) {
  // The casts are necessary to reassure C++ compilers
  cuda_enter = (void (*)(gpucontext *))gpuarray_get_extension("cuda_enter");
  cuda_exit = (void (*)(gpucontext *))gpuarray_get_extension("cuda_exit");
  cuda_make_ctx = (gpucontext *(*)(CUcontext, int))gpuarray_get_extension("cuda_make_ctx");
  cuda_get_stream = (CUstream (*)(void *))gpuarray_get_extension("cuda_get_stream");
  cuda_make_buf = (gpudata *(*)(void *, CUdeviceptr, size_t))gpuarray_get_extension("cuda_make_buf");
  cuda_get_ptr = (CUdeviceptr (*)(gpudata *))gpuarray_get_extension("cuda_get_ptr");
  cuda_get_sz = (size_t (*)(gpudata *))gpuarray_get_extension("cuda_get_sz");
  cuda_wait = (int (*)(gpudata *, int))gpuarray_get_extension("cuda_wait");
  cuda_record = (int (*)(gpudata *, int))gpuarray_get_extension("cuda_record");
}

#ifdef __cplusplus
}
#endif

#endif
