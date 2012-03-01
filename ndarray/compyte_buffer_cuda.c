#ifdef __linux__
#define _GNU_SOURCE
#endif
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/wait.h>

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>

#ifdef __APPLE__

#include <CUDA/cuda.h>

#else

#include <cuda.h>

#endif

#ifdef __linux__
size_t strlcpy(char *dst, const char *src, size_t siz);
size_t strlcat(char *dst, const char *src, size_t siz);
#endif

#include "compyte_buffer.h"
#include "compyte_util.h"

struct _gpudata {
    CUdeviceptr ptr;
    size_t sz;
};

static inline size_t gdata_size(gpudata *b) {
    return b->sz & SSIZE_MAX;
}

static inline int gdata_canfree(gpudata *b) {
    return (b->sz & ~SSIZE_MAX) ? 0 : 1;
}

static inline void gdata_setfree(gpudata *b) {
    b->sz |= ~SSIZE_MAX;
}

static inline void gdata_setsize(gpudata *b, size_t s) {
    b->sz = (b->sz & ~SSIZE_MAX) | s;
}

struct _gpukernel {
    CUmodule m;
    CUfunction k;
    void **args;
    unsigned int argcount;
#if CUDA_VERSION < 4000
    size_t *szs;
#endif
};


#define FAIL(v, e) { if (ret) *ret = e; return v; }
#define CHKFAIL(v) if (err != CUDA_SUCCESS) FAIL(v, GA_IMPL_ERROR)

static CUresult err;

static const char *get_error_string(CUresult err) {
    switch (err) {
    case CUDA_SUCCESS:                 return "Success!";
    case CUDA_ERROR_INVALID_VALUE:     return "Invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY:     return "Out of host memory";
    case CUDA_ERROR_NOT_INITIALIZED:   return "API not initialized";
    case CUDA_ERROR_DEINITIALIZED:     return "Driver is shutting down";
    case CUDA_ERROR_NO_DEVICE:         return "No CUDA devices avaiable";
    case CUDA_ERROR_INVALID_DEVICE:    return "Invalid device ordinal";
    case CUDA_ERROR_INVALID_IMAGE:     return "Invalid module image";
    case CUDA_ERROR_INVALID_CONTEXT:   return "No context bound to current thread or invalid context parameter";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "(deprecated) Context is already current";
    case CUDA_ERROR_MAP_FAILED:        return "Map or register operation failed";
    case CUDA_ERROR_UNMAP_FAILED:      return "Unmap of unregister operation failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED:   return "Array is currently mapped";
    case CUDA_ERROR_ALREADY_MAPPED:    return "Resource is already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "No kernel image suitable for device";
    case CUDA_ERROR_ALREADY_ACQUIRED:  return "Resource has already been acquired";
    case CUDA_ERROR_NOT_MAPPED:        return "Resource is not mapped";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "Resource cannot be accessed as array";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "Resource cannot be accessed as pointer";
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "Uncorrectable ECC error";
    case CUDA_ERROR_UNSUPPORTED_LIMIT: return "Limit not supported by device";
    case CUDA_ERROR_INVALID_SOURCE:    return "Invalid kernel source";
    case CUDA_ERROR_FILE_NOT_FOUND:    return "File was not found";
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "Could not resolve link to shared object";
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: return "Initialization of shared object failed";
    case CUDA_ERROR_OPERATING_SYSTEM:  return "OS call failed";
    case CUDA_ERROR_INVALID_HANDLE:    return "Invalid resource handle";
    case CUDA_ERROR_NOT_FOUND:         return "Symbol not found";
    case CUDA_ERROR_NOT_READY:         return "Previous asynchronous operation is still running";
    case CUDA_ERROR_LAUNCH_FAILED:     return "Kernel code raised an exception and destroyed the context";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "Not enough resource to launch kernel (or passed wrong arguments)";
    case CUDA_ERROR_LAUNCH_TIMEOUT:    return "Kernel took too long to execute";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "Kernel launch uses incompatible texture mode";
#if CUDA_VERSION >= 4000
    case CUDA_ERROR_PROFILER_DISABLED: return "Profiler is disabled";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "Profiler is not initialized";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "Profiler has already started";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "Profiler has already stopped";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "Context is already bound to another thread";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "Peer access already enabled";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "Peer access not enabled";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "Primary context already initialized";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "Context has been destroyed (or not yet initialized)";
    case CUDA_ERROR_ASSERT:            return "Kernel trigged an assert and destroyed the context";
    case CUDA_ERROR_TOO_MANY_PEERS:    return "Not enough ressoures to enable peer access";
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "Memory range already registered";
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: return "Memory range is not registered";
#endif
    case CUDA_ERROR_UNKNOWN:           return "Unknown internal error";
    default: return "Unknown error code";
    }
}

static void *cuda_init(int ord, int *ret) {
    CUdevice dev;
    CUcontext ctx;

    err = cuInit(0);
    CHKFAIL(NULL);
    err = cuDeviceGet(&dev, ord);
    CHKFAIL(NULL);
    err = cuCtxCreate(&ctx, CU_CTX_SCHED_AUTO, dev);
    CHKFAIL(NULL);
    return ctx;
}

static gpudata *cuda_alloc(void *ctx /* IGNORED */, size_t size, int *ret) {
    gpudata *res;

    res = malloc(sizeof(*res));
    if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
    
    res->sz = size;
    
    err = cuMemAlloc(&res->ptr, size);
    if (err != CUDA_SUCCESS) {
        free(res);
        FAIL(NULL, GA_IMPL_ERROR);
    }
    return res;
}

static gpudata *cuda_dup(gpudata *b, int *ret) {
    gpudata *res;
    res = malloc(sizeof(*res));
    if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
    
    res->ptr = b->ptr;
    res->sz = b->sz;
    gdata_setfree(res);
    return res;
}

static void cuda_free(gpudata *d) {
    if (gdata_canfree(d))
        err = cuMemFree(d->ptr);
    free(d);
}

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
    return ((a->ptr <= b->ptr && a->ptr + gdata_size(a) > b->ptr) ||
            (b->ptr <= a->ptr && b->ptr + gdata_size(b) > a->ptr));
}

static int cuda_move(gpudata *dst, gpudata *src)
{
    if (gdata_size(dst) != gdata_size(src))
        return GA_VALUE_ERROR;
    err = cuMemcpyDtoD(dst->ptr, src->ptr, gdata_size(dst));
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_read(void *dst, gpudata *src, size_t sz)
{
    if (sz > gdata_size(src))
        return GA_VALUE_ERROR;
    err = cuMemcpyDtoH(dst, src->ptr, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_write(gpudata *dst, const void *src, size_t sz)
{
    if (gdata_size(dst) != sz)
        return GA_VALUE_ERROR;
    err = cuMemcpyHtoD(dst->ptr, src, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, int data) {
    err = cuMemsetD8(dst->ptr, data, gdata_size(dst));
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_offset(gpudata *buf, ssize_t off) {
    /* XXX: this does not check for overflow */
    buf->ptr += off;
    gdata_setsize(buf, gdata_size(buf) - off);
    return GA_NO_ERROR;
}

/* This is a unix version, might need a windows one. */
static int call_compiler(char *fname, char *oname) {
    int sys_err;
    pid_t p;
    
    p = fork();
    if (p == 0) {
        /* Will need some way to specify arch (or detect it live) */
        execlp(CUDA_BIN_PATH "nvcc", "nvcc", "-x", "cu",
               "--cubin", fname, "-o", oname, NULL);
        exit(1);
    } else if (p == -1) {
        return GA_SYS_ERROR;
    }
    if (waitpid(p, &sys_err, 0) == -1)
        return GA_SYS_ERROR;
    if (WIFSIGNALED(sys_err) || WEXITSTATUS(sys_err) != 0) return GA_RUN_ERROR;
    return 0;
}

static const char INT_HEADERS[] = "#include <inttypes.h>\n";

static gpukernel *cuda_newkernel(void *ctx /* IGNORED */, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, int *ret) {
    char namebuf[MAXPATHLEN];
    char outbuf[MAXPATHLEN];
    char *tmpdir;
    int fd, sys_err;
    ssize_t s;
    struct iovec descr[count+1];
    gpukernel *res;
    unsigned int i;

    if (count == 0) FAIL(NULL, GA_VALUE_ERROR);

    descr[0].iov_base = (void *)INT_HEADERS;
    descr[0].iov_len = strlen(INT_HEADERS);

    if (lengths == NULL) {
        for (i = 0; i < count; i++) {
            descr[i+1].iov_base = (void *)strings[i];
            descr[i+1].iov_len = strlen(strings[i]);
        }
    } else {
        for (i = 0; i < count; i++) {
            descr[i+1].iov_base = (void *)strings[i];
            descr[i+1].iov_len = lengths[i]?lengths[i]:strlen(strings[i]);
        }
    }

    tmpdir = getenv("TMPDIR");
    if (tmpdir == NULL) tmpdir = "/tmp";

    strlcpy(namebuf, tmpdir, sizeof(namebuf));
    strlcat(namebuf, "/compyte.cuda.XXXXXXXX", sizeof(namebuf));

    fd = mkstemp(namebuf);
    if (fd == -1) FAIL(NULL, GA_SYS_ERROR);

    strlcpy(outbuf, namebuf, sizeof(outbuf));
    strlcat(outbuf, ".cubin", sizeof(outbuf));

    s = writev(fd, descr, count+1);
    /* fd is not non-blocking so should have complete write */
    if (s == -1) {
        close(fd);
        unlink(namebuf);
        FAIL(NULL, GA_SYS_ERROR);
    }
    sys_err = call_compiler(namebuf, outbuf);

    close(fd);
    unlink(namebuf);

    if (sys_err != GA_NO_ERROR) {
        unlink(outbuf);
        FAIL(NULL, sys_err);
    }

    res = malloc(sizeof(*res));
    if (res == NULL) {
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }
    res->args = NULL;
    res->argcount = 0;
#if CUDA_VERSION < 4000
    res->szs = NULL;
#endif

    err = cuModuleLoad(&res->m, outbuf);

    unlink(outbuf);

    if (err != CUDA_SUCCESS) {
        free(res);
        FAIL(NULL, GA_IMPL_ERROR);
    }

    if ((err = cuModuleGetFunction(&res->k, res->m, fname)) != CUDA_SUCCESS) {
        cuModuleUnload(res->m);
        free(res);
        FAIL(NULL, GA_IMPL_ERROR);
    }

    return res;
}

static void cuda_freekernel(gpukernel *k) {
    unsigned int i;
    for (i = 0; i < k->argcount; i++)
        free(k->args[i]);
    free(k->args);
#if CUDA_VERSION < 4000
    free(k->szs);
#endif
    cuModuleUnload(k->m);
    free(k);
}

static int cuda_setkernelarg(gpukernel *k, unsigned int index, size_t sz,
                             const void *val) {
    void *tmp;
    if (index >= k->argcount) {
        tmp = calloc(index+1, sizeof(void *));
        if (tmp == NULL) return GA_MEMORY_ERROR;
        bcopy(k->args, tmp, sizeof(void *)*k->argcount);
        free(k->args);
        k->args = (void **)tmp;
#if CUDA_VERSION < 4000
        tmp = calloc(index+1, sizeof(size_t));
        if (tmp == NULL) return GA_MEMORY_ERROR;
        bcopy(k->szs, tmp, sizeof(size_t)*k->argcount);
        free(k->szs);
        k->szs = (size_t *)tmp;
#endif
        k->argcount = index+1;
    }
    tmp = malloc(sz);
    if (tmp == NULL) return GA_MEMORY_ERROR;
    bcopy(val, tmp, sz);
    k->args[index] = tmp;
#if CUDA_VERSION < 4000
    k->szs[index] = sz;
#endif
    return GA_NO_ERROR;
}

static int cuda_setkernelargbuf(gpukernel *k, unsigned int index, gpudata *b) {
    return cuda_setkernelarg(k, index, sizeof(void *), &b->ptr);
}

static int cuda_callkernel(gpukernel *k, unsigned int gx, unsigned int gy,
                           unsigned int gz, unsigned int bx, unsigned int by,
                           unsigned int bz) {
#if CUDA_VERSION >= 4000
    err = cuLaunchKernel(k->k, gx, gy, gz, bx, by, bz, 0, NULL, k->args, NULL);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
#else
    size_t total = 0;
    unsigned int i;
    for (i = 0; i < k->argcount; i++)
        total += k->szs[i];
    err = cuParamSetSize(k->k, total);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    total = 0;
    for (i = 0; i < k->argcount; i++) {
        err = cuParamSetv(k->k,(int)total,k->args[i],(unsigned int)k->szs[i]);
        if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
        total += k->szs[i];
    }
    err = cuFuncSetBlockShape(k->k, (int)bx, (int)by, (int)bz);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuLaunchGrid(k->k, (int)gx, (int)gz*gy);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
#endif
    err = cuCtxSynchronize();
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static const char ELEM_HEADER[] = "#define DTYPEA %s\n"
    "#define DTYPEB %s\n"
    "extern \"C\" {"
    "__global__ void elemk(const DTYPEA *a_data, DTYPEB *b_data) {"
    "const int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    "const int numThreads = blockDim.x * gridDim.x;"
    "for (int i = idx; i < %zu; i += numThreads) {"
    "const DTYPEA *a = a_data;"
    "DTYPEB *b = b_data;";

static const char ELEM_FOOTER[] = "}}}\n";

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
    unsigned int i;

    for (i = 0; i < a_nd; i++) {
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
        /* XXX: does cuda does C-style pointer manip? */
        if (compyte_elem_perdim(strs, &count, a_nd, a_dims, a_str, "a",
                                compyte_get_elsize(intype)) == -1)
            goto fail;
        if (compyte_elem_perdim(strs, &count, b_nd, b_dims, b_str, "b",
                                compyte_get_elsize(outtype)) == -1)
            goto fail;

        if (asprintf(&strs[count], "b[0] %s a[0];", op) == -1)
            goto fail;
        count++;
    }

    strs[count] = strdup(ELEM_FOOTER);
    if (strs[count] == NULL) goto fail;
    count++;
    
    assert(count < (sizeof(strs)/sizeof(strs[0])));

    k = cuda_newkernel(NULL, count, (const char **)strs, NULL, "elemk", &res);
    if (k == NULL) goto fail;
    res = cuda_setkernelargbuf(k, 0, input);
    if (res != GA_NO_ERROR) goto failk;
    res = cuda_setkernelargbuf(k, 1, output);
    if (res != GA_NO_ERROR) goto failk;

    /* XXX: Revise this crappy block/grid assigment */
    unsigned int gx, bx;
    bx = xmin(32, nEls);
    gx = xmin((nEls/bx)+((nEls % bx != 0)?1:0), 60);
    if (bx*gx < nEls)
        bx = xmin(nEls/gx, 512);

    res = cuda_callkernel(k, gx, 1, 1, bx, 1, 1);

failk:
    cuda_freekernel(k);
fail:
    for (i = 0; i < count; i++) {
        free(strs[i]);
    }
    return res;
}

static const char *cuda_error(void) {
    return get_error_string(err);
}

compyte_buffer_ops cuda_ops = {cuda_init,
                               cuda_alloc,
                               cuda_dup,
                               cuda_free,
                               cuda_share,
                               cuda_move,
                               cuda_read,
                               cuda_write,
                               cuda_memset,
                               cuda_offset,
                               cuda_newkernel,
                               cuda_freekernel,
                               cuda_setkernelarg,
                               cuda_setkernelargbuf,
                               cuda_callkernel,
                               cuda_elemwise,
                               cuda_error};

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
