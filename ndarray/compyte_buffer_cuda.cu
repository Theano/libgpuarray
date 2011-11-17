#include <stdlib.h>

#include "compyte_buffer.h"

#define CNDA_THREAD_SYNC cudaThreadSynchronize()

struct _gpudata {
    char *ptr;
};

static gpudata *cuda_alloc(void *ctx /* IGNORED */, size_t size)
{
    /* ctx is ignored since it is implied from the context stack */
    cudaError_t err;
    gpudata *res;
    res = (gpudata *)malloc(sizeof(*res));
    if (res == NULL) {
        return NULL;
    }
    err = cudaMalloc(&res->ptr, size);
    if (cudaSuccess != err) {
        return NULL;
    }
    return res;
}

static void cuda_free(gpudata *d) {
    cudaFree(d->ptr);
    free(d);
}

static int cuda_move(gpudata *dst, gpudata *src, size_t sz)
{
    cudaError_t err;
    err = cudaMemcpy(dst->ptr, src->ptr, sz, cudaMemcpyDeviceToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return GA_IMPL_ERROR;
    }
    return 0;
}

static int cuda_read(void *dst, gpudata *src, size_t sz)
{
    cudaError_t err;
    err = cudaMemcpy(dst, src->ptr, sz, cudaMemcpyDeviceToHost);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return GA_IMPL_ERROR;
    }
    return 0;
}

static int cuda_write(gpudata *dst, void *src, size_t sz)
{
    cudaError_t err;
    err = cudaMemcpy(dst->ptr, src, sz, cudaMemcpyHostToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return GA_IMPL_ERROR;
    }
    return 0;
}

static int cuda_memset(gpudata *dst, int data, size_t bytes)
{
    cudaError_t err;
    err = cudaMemset(dst->ptr, data, bytes);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return GA_IMPL_ERROR;
    }
    return 0;
}

static int cuda_offset(gpudata *buf, int off) {
    buf->ptr += off;
    return 0;
}

static const char *cuda_error(void)
{
    return cudaGetErrorString(cudaPeekAtLastError());
}

compyte_buffer_ops cuda_ops = {cuda_alloc, cuda_free, cuda_move, cuda_read, cuda_write, cuda_memset, cuda_offset, cuda_error};

/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  c-file-offsets:((innamespace . 0)(inline-open . 0))
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
