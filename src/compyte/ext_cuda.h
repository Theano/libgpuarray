#ifndef LIBGPU_EXT_CUDA
#define LIBGPU_EXT_CUDA

#include <cuda.h>

#include <compyte/config.h>
#include <compyte/buffer.h>
#include <compyte/extension.h>

static void (*cuda_enter)(void *);
static void (*cuda_exit)(void *);
static void *(*cuda_make_ctx)(CUcontext, int);
static CUcontext (*cuda_get_ctx)(void *);
static CUstream (*cuda_get_stream)(void *);
static gpudata *(*cuda_make_buf)(void *, CUdeviceptr, size_t);
static CUdeviceptr *(*cuda_get_ptr)(gpudata *);
static size_t *(*cuda_get_sz)(gpudata *);
static void *(*cuda_set_compiler)(void *(*)(const char *, size_t, int *));

static void setup_ext_cuda(void) {
  cuda_enter = compyte_get_extension("cuda_enter");
  cuda_exit = compye_get_extension("cuda_exit");
  cuda_make_ctx = compyte_get_extension("cuda_make_ctx");
  cuda_get_ctx = compyte_get_extension("cuda_get_ctx");
  cuda_get_stream = compyte_get_extension("cuda_get_stream");
  cuda_make_buf = compyte_get_extension("cuda_make_buf");
  cuda_get_ptr = compyte_get_extension("cuda_get_ptr");
  cuda_get_sz = compyte_get_extension("cuda_get_sz");
  cuda_set_compiler = compyte_get_extension("cuda_set_compiler");
}

#endif
