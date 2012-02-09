#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <cuda.h>

#include "compyte_buffer.h"
#include "compyte_util.h"

#define CNDA_THREAD_SYNC cudaThreadSynchronize()

struct _gpudata {
    char *ptr;
};

struct _gpukernel {
    CUmodule m;
    CUfunction k;
    void **args;
    unsigned int argcount;
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
    res->args = NULL;
    res->argcount = 0;
    
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
    for (unsigned int i = 0; i < k->argcount; i++)
        free(k->args[i]);
    free(k->args);
    cuModuleUnload(k->m);
    free(k);
}

static int cuda_setkernelarg(gpukernel *k, unsigned int index, size_t sz,
                             const void *val) {
    void *tmp;
    if (index > k->argcount) {
        tmp = calloc(index+1, sizeof(void *));
        if (tmp == NULL) return GA_MEMORY_ERROR;
        bcopy(k->args, tmp, sizeof(void *)*k->argcount);
        free(k->args);
        k->args = (void **)tmp;
        k->argcount = index+1;
    }
    tmp = malloc(sz);
    if (tmp == NULL) return GA_MEMORY_ERROR;
    bcopy(val, tmp, sz);
    k->args[index] = tmp;
    return GA_NO_ERROR;
}

static int cuda_setkernelargbuf(gpukernel *k, unsigned int index, gpudata *b) {
    return cuda_setkernelarg(k, index, sizeof(void *), &b->ptr);
}

static int cuda_callkernel(gpukernel *k, unsigned int gx, unsigned int gy,
                           unsigned int gz, unsigned int bx, unsigned int by,
                           unsigned int bz) {
    CUresult err;
    
    err = cuLaunchKernel(k->k, gx, gy, gz, bx, by, bz, 0, NULL, k->args, NULL);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static const char ELEM_HEADER[] = "#define DTYPEA %s\n"
    "#define DTYPEB %s\n"
    "__global__ void elemk(const DTYPEA *a_data, DTYPEB *b_data) {"
    "const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    "const int numThreads = blockDim.x * gridDim.x;"
    "for (int i = idx; i < %zu; i += numThreads) {"
    "const DTYPEA *a = a_data;"
    "DTYPEB *b = b_data;";

static const char ELEM_FOOTER[] = "}}\n";

static inline unsigned int xmin(unsigned long a, unsigned long b) {
    return (unsigned int)((a < b) ? a : b);
}

static int cuda_elemwise(gpudata *input, gpudata *output, int intype,
                         int outtype, const char *op, unsigned int a_nd,
                         const size_t *a_dims, const ssize_t *a_str,
                         unsigned int b_nd, const size_t *b_dims,
                         const ssize_t *b_str) {
    char *strs[64];
    unsigned int count = 0;
    int res = GA_SYS_ERROR;
    
    size_t nEls = 1;
    gpukernel *k;

    for (unsigned int i = 0; i < a_nd; i++) {
        nEls *= a_dims[i];
    }
    
    if (asprintf(&strs[count], ELEM_HEADER,
                 compyte_get_type(intype)->cuda_name,
                 compyte_get_type(outtype)->cuda_name,
                 nEls) == -1)
        goto fail;
    count++;

    if (0) { /* contiguous case */
        if (asprintf(&strs[count], "b[i] %s a[i];", op) == -1)
            goto fail;
        count++;
    } else {
        if (compyte_elem_perdim(strs, &count, a_nd, a_dims, a_str, "a") == -1)
            goto fail;
        if (compyte_elem_perdim(strs, &count, b_nd, b_dims, b_str, "b") == -1)
            goto fail;

        if (asprintf(&strs[count], "b[0] %s a[0];", op) == -1)
            goto fail;
        count++;
    }

    strs[count] = (char *)ELEM_FOOTER;
    count++;
    
    assert(count < (sizeof(strs)/sizeof(strs[0])));

    res = GA_IMPL_ERROR;
    k = cuda_newkernel(NULL, count, (const char **)strs, NULL, "elemk");
    if (k == NULL) goto fail;
    res = cuda_setkernelargbuf(k, 0, input);
    if (res != GA_NO_ERROR) goto fail;
    res = cuda_setkernelargbuf(k, 1, output);
    if (res != GA_NO_ERROR) goto fail;

    /* XXX: Revise this crappy block/grid assigment */
    unsigned int gx, bx;
    bx = xmin(32, nEls);
    gx = xmin((nEls/bx)+((nEls % bx != 0)?1:0), 60);
    if (bx*gx < nEls)
        bx = xmin(nEls/gx, 512);

    res = cuda_callkernel(k, gx, 1, 1, bx, 1, 1);

fail:
    for (unsigned int i = 0; i < count; i++) {
        free(strs[i]);
    }
    return res;
}

static const char *cuda_error(void)
{
    return cudaGetErrorString(cudaPeekAtLastError());
}

compyte_buffer_ops cuda_ops = {cuda_alloc, cuda_free, cuda_move, cuda_read, cuda_write, cuda_memset, cuda_offset, cuda_newkernel, cuda_freekernel, cuda_setkernelarg, cuda_setkernelargbuf, cuda_callkernel, cuda_elemwise, cuda_error};

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
