#define _CRT_SECURE_NO_WARNINGS
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

#ifdef _MSC_VER
#include <io.h>
#define read _read
#define write _write
#define close _close
#define unlink _unlink
#define fstat _fstat
#define stat _stat
#define open _open
#define strdup _strdup
#else
#include <unistd.h>
#endif


#ifdef __APPLE__

#include <CUDA/cuda.h>

#else

#include <cuda.h>

#endif

#include "compyte_buffer.h"
#include "compyte_util.h"

typedef struct {char c; CUdeviceptr x; } st_devptr;
#define DEVPTR_ALIGN (sizeof(st_devptr) - sizeof(CUdeviceptr))

static int DONTFREE_ADDR;
#define DONTFREE ((gpudata *)&DONTFREE_ADDR)

struct _gpudata {
    CUdeviceptr ptr;
    size_t sz;
    gpudata *base;
    unsigned int refcnt;
};

gpudata *cuda_make_buf(CUdeviceptr p, size_t sz) {
    gpudata *res;
    res = malloc(sizeof(*res));
    if (res == NULL) return NULL;

    res->ptr = p;
    res->sz = sz;
    res->base = DONTFREE;
    res->refcnt = 1;

    return res;
}

CUdeviceptr cuda_get_ptr(gpudata *g) { return g->ptr; }
size_t cuda_get_sz(gpudata *g) { return g->sz; }

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
    case CUDA_ERROR_INVALID_VALUE:     return "Invalid cuda value";
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
    
    if (size != 0) {
        err = cuMemAlloc(&res->ptr, size);
        if (err != CUDA_SUCCESS) {
            free(res);
            FAIL(NULL, GA_IMPL_ERROR);
        }
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
    if (res->base != DONTFREE)
        res->base->refcnt += 1;
    return res;
}

static void cuda_free(gpudata *d) {
    d->refcnt -= 1;
    if (d->refcnt == 0) {
        if (d->base == NULL) {
            if (d->sz != 0)
                cuMemFree(d->ptr);
        }
        else if (d->base != DONTFREE)
            cuda_free(d->base);
        free(d);
    }
}

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
    return (a->sz != 0 && b->sz != 0 &&
            ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
             (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr)));
}

static int cuda_move(gpudata *dst, gpudata *src, size_t sz)
{
    if (dst->sz < sz || src->sz < sz)
        return GA_VALUE_ERROR;
    if (sz == 0) return GA_NO_ERROR;

    err = cuMemcpyDtoD(dst->ptr, src->ptr, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_read(void *dst, gpudata *src, size_t sz)
{
    if (src->sz < sz)
        return GA_VALUE_ERROR;
    if (sz == 0) return GA_NO_ERROR;

    err = cuMemcpyDtoH(dst, src->ptr, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_write(gpudata *dst, const void *src, size_t sz)
{
    if (dst->sz < sz)
        return GA_VALUE_ERROR;
    if (sz == 0) return GA_NO_ERROR;

    err = cuMemcpyHtoD(dst->ptr, src, sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, int data) {
    if (dst->sz == 0) return GA_NO_ERROR;

    err = cuMemsetD8(dst->ptr, data, dst->sz);
    if (err != CUDA_SUCCESS) {
        return GA_IMPL_ERROR;
    }
    return GA_NO_ERROR;
}

static int cuda_offset(gpudata *buf, ssize_t off) {
    /* This only check that you don't wrap around through the bottom */
    if (off > 0 && (size_t)off > buf->sz) return GA_VALUE_ERROR;
    buf->ptr += off;
    buf->sz -= off;
    return GA_NO_ERROR;
}

static const char *detect_arch(int *ret) {
    CUdevice dev;
    int major, minor;
    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) FAIL(NULL, GA_IMPL_ERROR);
    err = cuDeviceComputeCapability(&major, &minor, dev);
    if (err != CUDA_SUCCESS) FAIL(NULL, GA_IMPL_ERROR);
    switch (major) {
    case 1:
        switch (minor) {
        case 0:
            return "sm_10";
        case 1:
            return "sm_11";
        case 2:
            return "sm_12";
        default:
            return "sm_13";
        }
    case 2:
        switch (minor) {
        case 0:
            return "sm_20";
        default:
            return "sm_21";
        }
    default:
        return "sm_30";
    }
}

static const char *TMP_VAR_NAMES[] = {"COMPYTE_TMPDIR", "TMPDIR", "TEMP", "TMP"};

static void *call_compiler_impl(const char *src, size_t len, int *ret) {
    char namebuf[PATH_MAX];
    char outbuf[PATH_MAX];
    char *tmpdir;
    const char *arch_arg;
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

    arch_arg = detect_arch(&res);
    if (arch_arg == NULL) FAIL(NULL, res);

    for (i = 0; i < sizeof(TMP_VAR_NAMES)/sizeof(TMP_VAR_NAMES[0]); i++) {
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
    if (p == -1) {
        unlink(namebuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    /* We need to wait until after the waitpid for the unlink because otherwise
       we might delete the input file before nvcc is finished with it. */
    if (waitpid(p, &sys_err, 0) == -1) {
        unlink(namebuf);
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    } else {
        unlink(namebuf);
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

void *(*cuda_call_compiler)(const char *src, size_t len, int *ret) = call_compiler_impl;

static gpukernel *cuda_newkernel(void *ctx /* IGNORED */, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, int *ret) {
    struct iovec *descr;
    char *buf;
    char *p;
    gpukernel *res;
    size_t tot_len;
    unsigned int i;
    int ptx_mode = 0;

    if (count == 0) FAIL(NULL, GA_VALUE_ERROR);
    
    descr = calloc(count+1, sizeof(*descr));
    if (descr == NULL) FAIL(NULL, GA_SYS_ERROR);

    if ((!lengths || lengths[0] > 3) && strcmp(strings[0], "PTX") == 0) {
        ptx_mode = 1;
        descr[0].iov_base = "";
        descr[0].iov_len = 0;
        /* skip the first string (since it's a marker) */
        strings++; count--;
    } else {
        descr[0].iov_base = (void *)CUDA_HEAD;
        descr[0].iov_len = strlen(CUDA_HEAD);
    }
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

    if (ptx_mode) tot_len += 1;

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

    if (ptx_mode) p[0] = '\0';

    free(descr);

    if (ptx_mode) {
        p = buf;
    } else {
        p = cuda_call_compiler(buf, tot_len, ret);
        free(buf);
        if (p == NULL)
            return NULL;
    }

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

static int do_sched(gpukernel *k, size_t n, unsigned int *bc,
                    unsigned int *tpb) {
    CUdevice dev;
    int min_t;
    int max_t;
    int max_b;
    unsigned int grp;

    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuDeviceGetAttribute(&min_t, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuDeviceGetAttribute(&max_t, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                               dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuDeviceGetAttribute(&max_b, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                               dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;

    if (n < (unsigned)(max_b * min_t)) {
        *bc = (n + min_t - 1) / min_t;
        *tpb = min_t;
    } else if (n < (unsigned)(max_b * max_t)) {
        *bc = max_b;
        grp = (n + min_t - 1) / min_t;
        *tpb = ((grp + max_b - 1) / max_b) * min_t;
/*
 *bc = (n + max_t - 1) / max_t;
 *tpb = max_t;
*/
    } else if (n == (unsigned)(max_b * max_t)) {
        *bc = max_b;
        *tpb = max_t;
    } else {
        /* Too many elements */
        return GA_VALUE_ERROR;
    }
    return GA_NO_ERROR;
}

#define ALIGN_UP(offset, align) ((offset) + (align) - 1) & ~((align) - 1)

static int cuda_callkernel(gpukernel *k, size_t n) {
    int res;
    unsigned int block_count;
    unsigned int threads_per_block;

    res = do_sched(k, n, &block_count, &threads_per_block);
    if (res != GA_NO_ERROR)
        return res;

#if CUDA_VERSION >= 4000
    err = cuLaunchKernel(k->k, block_count, 1, 1, threads_per_block, 1, 1,
                         0, NULL, k->args, NULL);
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
    err = cuFuncSetBlockShape(k->k, threads_per_block, 1, 1);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = cuLaunchGrid(k->k, block_count, 1);
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
    "const char *a_p = (const char *)a_data;"
    "char *b_p = (char *)b_data;";

static const char ELEM_FOOTER[] =
    "const DTYPEA *a = (const DTYPEA *)a_p;"
    "DTYPEB *b = (DTYPEB *)b_p;"
    "b[0] = a[0];}}}\n";

#ifdef CUDA_PTX
static const char ELEM_HEADER_PTX[] = ".version 3.0\n.target %s\n\n"
    ".entry elemk (\n"
    ".param .u%u a_data,\n"
    ".param .u%u b_data ) {\n"
    ".reg .u16 rh1, rh2;\n"
    ".reg .u32 numThreads;\n"
    ".reg .u32 i;\n"
    ".reg .u32 a_pi, b_pi;\n"
    ".reg .u32 a_p, b_p;\n"
    ".reg .u32 r1;\n"
    ".reg .u%u rp1, rp2;\n"
    ".reg .%s tmpa;\n"
    ".reg .%s tmpb;\n"
    ".reg .pred p;\n"
    "mov.u16 rh1, %%ntid.x;\n"
    "mov.u16 rh2, %%ctaid.x;\n"
    "mul.wide.u16 i, rh1, rh2;\n"
    "mov.u32 r1, %%tid.x;\n"
    "add.u32 i, i, r1;\n"
    "mov.u16 rh2, %%nctaid.x;\n"
    "mul.wide.u16 numThreads, rh2, rh1;\n"
    "setp.ge.u32 p, i, %" SPREFIX "uU;\n"
    "@p bra $end;\n"
    "$loop_begin:\n"
    "mov.u32 a_p, 0U;\n"
    "mov.u32 b_p, 0U;\n";

static inline ssize_t ssabs(ssize_t v) {
    return (v < 0 ? -v : v);
}

static int cuda_perdim_ptx(char *strs[], unsigned int *count, unsigned int nd,
                           const size_t *dims, const ssize_t *str,
                           const char *id) {
    int i;

    if (nd > 0) {
        if (asprintf(&strs[*count], "mov.u32 %si, i;\n", id) == -1)
            return -1;
        (*count)++;

        for (i = nd-1; i > 0; i--) {
            if (asprintf(&strs[*count], "rem.u32 r1, %si, %" SPREFIX "uU;\n"
                         "mul.lo.u32 r1, r1, %" SPREFIX "d;\n"
                         "%s.u32 %s, %s, r1;\n"
                         "div.u32 %si, %si, %" SPREFIX "uU;\n",
                         id, dims[i], ssabs(str[i]),
                         (str[i] < 0 ? "sub" : "add"), id, id, id, id,
                         dims[i]) == -1)
                return -1;
            (*count)++;
        }

        if (asprintf(&strs[*count], "mul.lo.u32 r1, %si, %" SPREFIX "d;\n"
                     "%s.u32 %s, %s, r1;\n", id, ssabs(str[0]),
                     (str[0] < 0 ? "sub" : "add"), id, id) == -1)
            return -1;
        (*count)++;
    }
    return 0;
}

static const char ELEM_FOOTER_PTX[] = "add.u32 i, i, numThreads;\n"
    "setp.lt.u32 p, i, %" SPREFIX "uU;\n"
    "@p bra $loop_begin;\n"
    "$end:\n"
    "}\n";

static inline const char *map_t(int typecode) {
    switch (typecode) {
    case GA_BYTE:
        return "s8";
    case GA_BOOL:
    case GA_UBYTE:
        return "u8";
    case GA_SHORT:
        return "s16";
    case GA_USHORT:
        return "u16";
    case GA_INT:
        return "s32";
    case GA_UINT:
        return "u32";
    case GA_LONG:
        return "s64";
    case GA_ULONG:
        return "u64";
    case GA_FLOAT:
        return "f32";
    case GA_DOUBLE:
        return "f64";
    case GA_HALF:
        return "f16";
    default:
        return NULL;
    }
}

#endif

static inline unsigned int xmin(unsigned long a, unsigned long b) {
    return (unsigned int)((a < b) ? a : b);
}

static int cuda_extcopy(gpudata *input, gpudata *output, int intype,
                        int outtype, unsigned int a_nd,
                        const size_t *a_dims, const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str) {
    char *strs[64];
    unsigned int count = 0;
    int res = GA_SYS_ERROR;
    
    size_t nEls = 1;
    gpukernel *k;
    unsigned int i;

#ifdef CUDA_PTX
    unsigned int bits = sizeof(void *)*8;
    const char *in_t;
    const char *out_t;
    const char *arch;

    in_t = map_t(intype);
    out_t = map_t(outtype);
    if (in_t == NULL || out_t == NULL) return GA_UNSUPPORTED_ERROR;
    arch = detect_arch(&res);
    if (arch == NULL) return res;
#define compyte_elem_perdim cuda_perdim_ptx
#endif

    for (i = 0; i < a_nd; i++) {
        nEls *= a_dims[i];
    }

    if (nEls == 0) return GA_NO_ERROR;

#ifdef CUDA_PTX
    // This is a marker to let the newkernel function know that we are
    // providing a PTX kernel rather than a C one.
    strs[0] = strdup("PTX");
    if (strs[0] == NULL) goto fail;
    count = 1;

    if (asprintf(&strs[count], ELEM_HEADER_PTX, arch,
                 bits, bits, bits, in_t, out_t, nEls) == -1)
#else
    if (asprintf(&strs[count], ELEM_HEADER,
                 compyte_get_type(intype)->cuda_name,
                 compyte_get_type(outtype)->cuda_name,
                 nEls) == -1)
#endif
        goto fail;
    count++;

    if (compyte_elem_perdim(strs, &count, a_nd, a_dims, a_str, "a_p") == -1)
        goto fail;
    if (compyte_elem_perdim(strs, &count, b_nd, b_dims, b_str, "b_p") == -1)
        goto fail;

#ifdef CUDA_PTX
    if (asprintf(&strs[count], "ld.param.u%u rp1, [a_data];\n"
                 "cvt.u32.u32 rp2, a_p;\n"
                 "add.u%u rp1, rp1, rp2;\n"
                 "ld.global.%s tmpa, [rp1];\n"
                 "cvt.%s.%s tmpb, tmpa;\n"
                 "ld.param.u%u rp1, [b_data];\n"
                 "cvt.u32.u32 rp2, b_p;\n"
                 "add.u%u rp1, rp1, rp2;\n"
                 "st.global.%s [rp1], tmpb;\n", bits, bits, in_t,
                 out_t, in_t, bits, bits, out_t) == -1)
        goto fail;
    count++;

    if (asprintf(&strs[count], ELEM_FOOTER_PTX, nEls) == -1)
        goto fail;
    count++;
#else
    strs[count] = strdup(ELEM_FOOTER);
    if (strs[count] == NULL)
        goto fail;
    count++;
#endif

    assert(count < (sizeof(strs)/sizeof(strs[0])));

    k = cuda_newkernel(NULL, count, (const char **)strs, NULL, "elemk", &res);
    if (k == NULL) goto fail;
    res = cuda_setkernelargbuf(k, 0, input);
    if (res != GA_NO_ERROR) goto failk;
    res = cuda_setkernelargbuf(k, 1, output);
    if (res != GA_NO_ERROR) goto failk;

    res = cuda_callkernel(k, nEls);

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

static const char CUDA_PREAMBLE[] =
    "#define local_barrier() __syncthreads()\n"
    "#define WITHIN_KERNEL __device__\n"
    "#define KERNEL extern \"C\" __global__\n"
    "#define GLOBAL_MEM /* empty */\n"
    "#define LOCAL_MEM __shared__\n"
    "#define LOCAL_MEM_ARG /* empty */\n"
    "#define REQD_WG_SIZE(X,Y,Z) __launch_bounds__(X*Y*Z, 1)\n"
    "#define LID_0 threadIdx.x\n"
    "#define LID_1 threadIdx.y\n"
    "#define LID_2 threadIdx.z\n"
    "#define LDIM_0 blockDim.x\n"
    "#define LDIM_1 blockDim.y\n"
    "#define LDIM_2 blockDim.z\n"
    "#define GID_0 blockIdx.x\n"
    "#define GID_1 blockIdx.y\n"
    "#define GID_2 blockIdx.z\n"
    "#define GDIM_0 gridDim.x\n"
    "#define GDIM_1 gridDim.y\n"
    "#define GDIM_2 gridDim.z\n";

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
                               cuda_extcopy,
                               cuda_error,
                               CUDA_PREAMBLE};

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
