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
static size_t (*cuda_get_sz)(gpudata *);
static int (*cuda_wait)(gpudata *, int);
static int (*cuda_record)(gpudata *, int);
static CUipcMemHandle (*cuda_get_ipc_handle)(gpudata *d);
static gpudata *(*cuda_open_ipc_handle)(gpucontext *c, CUipcMemHandle h,
                                        size_t sz);

static void setup_ext_cuda(void) {
  // The casts are necessary to reassure C++ compilers
  cuda_enter = (void (*)(gpucontext *))gpuarray_get_extension("cuda_enter");
  cuda_exit = (void (*)(gpucontext *))gpuarray_get_extension("cuda_exit");
  cuda_make_ctx = (gpucontext *(*)(CUcontext, int))gpuarray_get_extension("cuda_make_ctx");
  cuda_get_stream = (CUstream (*)(void *))gpuarray_get_extension("cuda_get_stream");
  cuda_make_buf = (gpudata *(*)(void *, CUdeviceptr, size_t))gpuarray_get_extension("cuda_make_buf");
  cuda_get_sz = (size_t (*)(gpudata *))gpuarray_get_extension("cuda_get_sz");
  cuda_wait = (int (*)(gpudata *, int))gpuarray_get_extension("cuda_wait");
  cuda_record = (int (*)(gpudata *, int))gpuarray_get_extension("cuda_record");
  cuda_get_ipc_handle = (CUipcMemHandle (*)(gpudata *))gpuarray_get_extension("cuda_get_ipc_handle");
  cuda_open_ipc_handle = (gpudata *(*)(gpucontext *c, CUipcMemHandle h, size_t sz))gpuarray_get_extension("cuda_open_ipc_handle");
}

#ifdef __cplusplus
}
#endif

#endif
