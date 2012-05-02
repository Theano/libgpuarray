#include "compyte_compat.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <limits.h>

#ifdef _WIN32
#include <process.h>
/* I am really tired of hunting through online docs
 * to find where the define is.  256 seem to be the
 * concencus for the value so there it is.
 */
#define PATH_MAX 256
#else
#include <sys/param.h>
#include <sys/wait.h>
#endif

#ifdef __APPLE__

#include <CUDA/cuda.h>

#else

#include <cuda.h>

#endif

#include "compyte_buffer.h"
#include "compyte_util.h"

#ifdef _MSC_VER
#include <io.h>
#define read _read
#define write _write
#define close _close
#define unlink _unlink
#define fstat _fstat
#define stat _stat
#endif

typedef struct {char c; CUdeviceptr x; } st_devptr;
#define DEVPTR_ALIGN (sizeof(st_devptr) - sizeof(CUdeviceptr))

struct _gpudata {
    CUdeviceptr ptr;
    size_t sz;
    gpudata *base;
    unsigned int refcnt;
};

/* The total size of the arguments is limited to 256 bytes */
#define NUM_ARGS (256/sizeof(void*))

struct _gpukernel {
    CUmodule m;
    CUfunction k;
    void *args[NUM_ARGS];
#if CUDA_VERSION < 4000
    size_t types[NUM_ARGS];
#endif
    unsigned int argcount;
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
    res->base = NULL;
    res->refcnt = 1;
    
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
    res->base = b;
    if (res->base->base != NULL)
        res->base = res->base->base;
    res->refcnt = 1;
    b->refcnt += 1;
    return res;
}

static void cuda_free(gpudata *d) {
    d->refcnt -= 1;
    if (d->refcnt == 0) {
        if (d->base != NULL)
            cuda_free(d->base);
        else
            cuMemFree(d->ptr);
        free(d);
    }
}

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
    return ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
            (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr));
}

static int cuda_move(gpudata *dst, gpudata *src)
{
    if (dst->sz != src->sz)
        return GA_VALUE_ERROR;
    err = cuMemcpyDtoD(dst->ptr, src->ptr, dst->sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_read(void *dst, gpudata *src, size_t sz)
{
    if (sz > src->sz)
        return GA_VALUE_ERROR;
    err = cuMemcpyDtoH(dst, src->ptr, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_write(gpudata *dst, const void *src, size_t sz)
{
    if (dst->sz != sz)
        return GA_VALUE_ERROR;
    err = cuMemcpyHtoD(dst->ptr, src, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, int data) {
    err = cuMemsetD8(dst->ptr, data, dst->sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_offset(gpudata *buf, ssize_t off) {
    /* XXX: this does not check for overflow */
    buf->ptr += off;
    buf->sz -= off;
    return GA_NO_ERROR;
}

static const char *arch_arg = NULL;

static int detect_arch(void) {
    CUdevice dev;
    int major, minor;
    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuDeviceComputeCapability(&major, &minor, dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    switch (major) {
    case 1:
        switch (minor) {
        case 0:
            arch_arg = "sm_10";
            break;
        case 1:
            arch_arg = "sm_11";
            break;
        case 2:
            arch_arg = "sm_12";
            break;
        case 3:
        default:
            arch_arg = "sm_13";
        }
        break;
    case 2:
        switch (minor) {
        case 0:
            arch_arg = "sm_20";
            break;
        case 1:
            arch_arg = "sm_21";
            break;
        case 2:
            arch_arg = "sm_22";
            break;
        case 3:
        default:
            arch_arg = "sm_23";
        }
        break;
    default:
        arch_arg = "sm_23";
    }
    return GA_NO_ERROR;
}

void *call_compiler(const char *src, size_t len, int *ret);

static const char *TMP_VAR_NAMES[] = {"COMPYTE_TMPDIR", "TMPDIR", "TEMP", "TMP"};

void *call_compiler_impl(const char *src, size_t len, int *ret) {
    char namebuf[PATH_MAX];
    char outbuf[PATH_MAX];
    char *tmpdir;
    struct stat st;
    ssize_t s;
#ifndef _WIN32
    pid_t p;
#endif
    unsigned int i;
    int sys_err;
    int fd;
    char *buf;
    int res;

    if (arch_arg == NULL) {
        res = detect_arch();
        if (res != GA_NO_ERROR) FAIL(NULL, res);
        assert(arch_arg != NULL);
    }

    for (i = 0; i < sizeof(TMP_VAR_NAMES); i++) {
        tmpdir = getenv(TMP_VAR_NAMES[i]);
        if (tmpdir != NULL) break;
    }
    if (tmpdir == NULL) {
#ifdef _WIN32
            tmpdir = ".";
#else
            tmpdir = "/tmp";
#endif
    }

    strlcpy(namebuf, tmpdir, sizeof(namebuf));
    strlcat(namebuf, "/compyte.cuda.XXXXXXXX", sizeof(namebuf));

    fd = mkstemp(namebuf);
    if (fd == -1) FAIL(NULL, GA_SYS_ERROR);

    strlcpy(outbuf, namebuf, sizeof(outbuf));
    strlcat(outbuf, ".cubin", sizeof(outbuf));

    s = write(fd, src, len);
    close(fd);
    /* fd is not non-blocking; should have complete write */
    if (s == -1) {
        unlink(namebuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    /* This block executes nvcc on the written-out file */
#define NVCC_ARGS "nvcc", "-arch", arch_arg, "-x", "cu", \
      "--cubin", namebuf, "-o", outbuf
#ifdef _WIN32
    sys_err = _spawnlp(_P_WAIT, NVCC_BIN, NVCC_ARGS, NULL);
    unlink(namebuf);
    if (sys_err == -1) FAIL(NULL, GA_SYS_ERROR);
    if (sys_err != 0) FAIL(NULL, GA_RUN_ERROR);
#else
    p = fork();
    if (p == 0) {
        execlp(NVCC_BIN, NVCC_ARGS, NULL);
        exit(1);
    }
    unlink(namebuf);
    if (p == -1)
        FAIL(NULL, GA_SYS_ERROR);

    if (waitpid(p, &sys_err, 0) == -1) {
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    if (WIFSIGNALED(sys_err) || WEXITSTATUS(sys_err) != 0) {
        unlink(outbuf);
        FAIL(NULL, GA_RUN_ERROR);
    }
#endif

    fd = open(outbuf, O_RDONLY);
    if (fd == -1) {
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    if (fstat(fd, &st) == -1) {
        close(fd);
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    buf = malloc(st.st_size);
    if (buf == NULL) {
        close(fd);
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    s = read(fd, buf, st.st_size);
    close(fd);
    unlink(outbuf);
    /* fd is not non-blocking; should have complete read */
    if (s == -1) {
        free(buf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    return buf;
}

static gpukernel *cuda_newkernel(void *ctx /* IGNORED */, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, int *ret) {
    struct iovec *descr;
    char *buf;
    char *p;
    gpukernel *res;
    size_t tot_len;
    unsigned int i;

    if (count == 0) FAIL(NULL, GA_VALUE_ERROR);
    
    descr = calloc(count+1, sizeof(*descr));
    if (descr == NULL) FAIL(NULL, GA_SYS_ERROR);

    descr[0].iov_base = (void *)CUDA_HEAD;
    descr[0].iov_len = strlen(CUDA_HEAD);
    tot_len = descr[0].iov_len;

    if (lengths == NULL) {
        for (i = 0; i < count; i++) {
            descr[i+1].iov_base = (void *)strings[i];
            descr[i+1].iov_len = strlen(strings[i]);
            tot_len += descr[i+1].iov_len;
        }
    } else {
        for (i = 0; i < count; i++) {
            descr[i+1].iov_base = (void *)strings[i];
            descr[i+1].iov_len = lengths[i]?lengths[i]:strlen(strings[i]);
            tot_len += descr[i+1].iov_len;
        }
    }

    buf = malloc(tot_len);
    if (buf == NULL) {
        free(descr);
        FAIL(NULL, GA_SYS_ERROR);
    }

    p = buf;
    for (i = 0; i < count+1; i++) {
        memcpy(p, descr[i].iov_base, descr[i].iov_len);
        p += descr[i].iov_len;
    }
    free(descr);

    p = call_compiler(buf, tot_len, ret);
    free(buf);
    if (p == NULL)
        return NULL;

    res = malloc(sizeof(*res));
    if (res == NULL)
        FAIL(NULL, GA_SYS_ERROR);

    memset(res, 0, sizeof(*res));

    err = cuModuleLoadData(&res->m, p);
    free(p);

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
    cuModuleUnload(k->m);
    free(k);
}

static int cuda_setkernelarg(gpukernel *k, unsigned int index, int typecode,
                             const void *val) {
    void *tmp;
    size_t sz;
    if (index >= NUM_ARGS) return GA_VALUE_ERROR;

    if (index >= k->argcount)
        k->argcount = index+1;

    /* Flag value (and a horrible abuse of reserved values) */
    if (typecode == GA_DELIM) {
        sz = sizeof(CUdeviceptr);
    } else {
        sz = compyte_get_elsize(typecode);
    }

    tmp = malloc(sz);
    if (tmp == NULL) return GA_MEMORY_ERROR;
    memcpy(tmp, val, sz);
    k->args[index] = tmp;
#if CUDA_VERSION < 4000
    k->types[index] = typecode;
#endif
    return GA_NO_ERROR;
}

static int cuda_setkernelargbuf(gpukernel *k, unsigned int index, gpudata *b) {
    return cuda_setkernelarg(k, index, GA_DELIM, &b->ptr);
}

#define ALIGN_UP(offset, align) ((offset) + (align) - 1) & ~((align) - 1)

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
    size_t align, sz;
    unsigned int i;
    for (i = 0; i < k->argcount; i++) {
        if (k->types[i] == GA_DELIM) {
            align = DEVPTR_ALIGN;
            sz = sizeof(CUdeviceptr);
        } else {
            align = compyte_get_type(k->types[i])->align;
            sz = compyte_get_elsize(k->types[i]);
        }
        total = ALIGN_UP(total, align);
        err = cuParamSetv(k->k, (int)total, k->args[i], (unsigned int)sz);
        if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
        total += sz;
    }
    err = cuParamSetSize(k->k, total);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
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
    "for (int i = idx; i < %" SPREFIX "u; i += numThreads) {"
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
    unsigned int gx, bx;

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
