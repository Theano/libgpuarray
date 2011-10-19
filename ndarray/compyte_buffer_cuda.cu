#include <stdlib.h>

#include "compyte_buffer.h"

#define CNDA_THREAD_SYNC cudaThreadSynchronize()

static gpudata *cuda_malloc(void *ctx /* IGNORED */, size_t size)
{
    /* ctx is ignored since it is implied from the context stack */
    cudaError_t err;
    char *res;
    err = cudaMalloc(&res, size);
    if (cudaSuccess != err) {
        return NULL;
    }
    return (gpudata *)res;
}

static void cuda_free(gpudata *d) {
    cudaFree(d);
}

static int cuda_move(gpudata *d_dst, size_t dst_offset,
                     gpudata *d_src, size_t src_offset, size_t sz)
{
    void *dst = (char *)d_dst + dst_offset;
    void *src = (char *)d_src + src_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_read(void *dst, gpudata *d_src, size_t src_offset, size_t sz)
{
    void *src = (char *)d_src + src_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_write(gpudata *d_dst, size_t dst_offset, void *src, size_t sz)
{
    void *dst = (char *)d_dst + dst_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_memset(gpudata *d_dst, size_t dst_offset, int data, size_t bytes)
{
    void *dst = (char *)d_dst + dst_offset;
    err = cudaMemset(dst, data, bytes);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static const char *cuda_error(void)
{
    /* Might want to use cudaPeekAtLastError() instead */
    return cudaGetErrorString(cudaGetLastError());
}

compyte_buffer_ops cuda_ops = {cuda_alloc, cuda_free, cuda_move, cuda_read, cuda_write, cuda_memset, cuda_error};

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
