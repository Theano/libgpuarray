#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <cuda.h>

#include "compyte_buffer.h"

#define CNDA_THREAD_SYNC cudaThreadSynchronize()

struct _gpudata {
    char *ptr;
};

struct _gpukernel {
    CUmodule m;
    CUfunction k;
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

/* This is a unix version, might need a windows one. */
static int call_compiler(char *fname, char *oname) {
    int err;
    pid_t p;
    
    p = fork();
    if (p == 0) {
        /* Will need some way to specify arch (or detect it live) */
        execlp("nvcc", "-xcu", "--cubin", fname, "-o", oname, NULL);
        exit(1);
    } else if (p == -1) {
        return GA_SYS_ERROR;
    }
    if (waitpid(p, &err, 0) == -1)
        return GA_SYS_ERROR;
    if (WIFSIGNALED(err) || WEXITSTATUS(err) != 0) return GA_SYS_ERROR;
    return 0;
}

static gpukernel *cuda_newkernel(void *ctx /* IGNORED */, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname) {
    char namebuf[MAXPATHLEN];
    char outbuf[MAXPATHLEN];
    char *tmpdir;
    int fd, err;
    ssize_t s;
    struct iovec descr[count];
    gpukernel *res;

    if (count == 0) return NULL;
    
    if (lengths == NULL) {
        for (unsigned int i = 0; i < count; i++) {
            descr[i].iov_base = (void *)strings[i];
            descr[i].iov_len = strlen(strings[i]);
        }
    } else {
        for (unsigned int i = 0; i < count; i++) {
            descr[i].iov_base = (void *)strings[i];
            descr[i].iov_len = lengths[i];
        }
    }
    
    tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL) tmpdir = "/tmp";
    
    strlcpy(namebuf, tmpdir, sizeof(namebuf));
    strlcat(namebuf, "/compyte.cuda.XXXXXXXX", sizeof(namebuf));

    fd = mkstemp(namebuf);
    if (fd == -1) return NULL;
    
    strlcpy(outbuf, namebuf, sizeof(outbuf));
    strlcat(outbuf, ".cubin", sizeof(outbuf));
    
    s = writev(fd, descr, count);
    /* fd is not non-blocking so should have complete write */
    if (s == -1) {
        close(fd);
        unlink(namebuf);
        return NULL;
    }
    err = call_compiler(namebuf, outbuf);

    close(fd);
    unlink(namebuf);

    if (err != GA_NO_ERROR) return NULL;
        
    res = (gpukernel *)malloc(sizeof(*res));
    if (res == NULL) return NULL;
    
    if (cuModuleLoad(&res->m, outbuf) != CUDA_SUCCESS) {
        free(res);
        return NULL;
    }

    if (cuModuleGetFunction(&res->k, res->m, fname) != CUDA_SUCCESS) {
        cuModuleUnload(res->m);
        free(res);
        return NULL;
    }
    return res;
}

static void cuda_freekernel(gpukernel *k) {
    cuModuleUnload(k->m);
    free(k);
}

static const char *cuda_error(void)
{
    return cudaGetErrorString(cudaPeekAtLastError());
}

compyte_buffer_ops cuda_ops = {cuda_alloc, cuda_free, cuda_move, cuda_read, cuda_write, cuda_memset, cuda_offset, cuda_newkernel, cuda_freekernel, cuda_error};

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
