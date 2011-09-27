#include "compyte_buffer.h"

#define CNDA_THREAD_SYNC cudaThreadSynchronize()

static cudaError_t err;

static void *cuda_malloc(size_t size)
{
    void *rval=NULL;
    err = cudaMalloc(&rval, size);
    if (cudaSuccess != err) {
        return NULL;
    }
    return rval;
}

static void cuda_free(void *ptr)
{
    err = cudaFree(ptr);
}

static int cuda_move(void * dst, size_t dst_offset,
                     void * src, size_t src_offset, size_t sz)
{
    dst += dst_offset;
    src += src_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_read(void * dst, void * src, size_t src_offset, size_t sz)
{
    src += src_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyDeviceToHost);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_write(void * dst, size_t dst_offset, void * src, size_t sz)
{
    dst += dst_offset;
    err = cudaMemcpy(dst, src, sz, cudaMemcpyHostToDevice);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static int cuda_memset(void *dst, size_t dst_offset, int data, size_t bytes)
{
    dst += dst_offset;
    err = cudaMemset(dst, data, bytes);
    CNDA_THREAD_SYNC;
    if (cudaSuccess != err) {
        return -1;
    }
    return 0;
}

static const char *cuda_error(void)
{
    return cudaGetErrorString(err);
}

static compyte_buffer_ops _cuda_ops = {cuda_alloc, cuda_free, cuda_move, cuda_read, cuda_write, cuda_memset, NULL, NULL, cuda_error};

compyte_buffer_ops *cuda_ops = &_cuda_ops;

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
