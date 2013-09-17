#define _CRT_SECURE_NO_WARNINGS
#include "private.h"

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
 * consensus for the value so there it is.
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

#include "compyte/buffer.h"
#include "compyte/util.h"
#include "compyte/error.h"
#include "compyte/extension.h"

typedef struct {char c; CUdeviceptr x; } st_devptr;
#define DEVPTR_ALIGN (sizeof(st_devptr) - sizeof(CUdeviceptr))

#define DONTFREE 0x1

static CUresult err;

typedef struct _cuda_context {
  CUcontext ctx;
  CUcontext old;
  CUresult err;
  CUstream s;
  unsigned int refcnt;
  int flags;
} cuda_context;

COMPYTE_LOCAL void *cuda_make_ctx(CUcontext ctx, int flags) {
  cuda_context *res;
  res = malloc(sizeof(*res));
  if (res == NULL)
    return NULL;
  res->ctx = ctx;
  res->err = CUDA_SUCCESS;
  res->refcnt = 1;
  res->flags = flags;
  err = cuStreamCreate(&res->s, 0);
  if (err != CUDA_SUCCESS) {
    free(res);
    return NULL;
  }
  return res;
}

static void cuda_free_ctx(cuda_context *ctx) {
  ctx->refcnt--;
  if (ctx->refcnt == 0) {
    cuStreamDestroy(ctx->s);
    if (!(ctx->flags & DONTFREE))
      cuCtxDestroy(ctx->ctx);
    free(ctx);
  }
}

COMPYTE_LOCAL CUcontext cuda_get_ctx(void *ctx) {
  return ((cuda_context *)ctx)->ctx;
}

COMPYTE_LOCAL CUstream cuda_get_stream(void *ctx) {
  return ((cuda_context *)ctx)->s;
}

static void cuda_enter(cuda_context *ctx) {
  cuCtxGetCurrent(&ctx->old);
  if (ctx->old != ctx->ctx)
    ctx->err = cuCtxSetCurrent(ctx->ctx);
  /* If no context was there in the first place, then we take over
     to avoid the set dance on the thread */
  if (ctx->old == NULL) ctx->old = ctx->ctx;
}

static void cuda_exit(cuda_context *ctx) {
  if (ctx->old != ctx->ctx)
    cuCtxSetCurrent(ctx->old);
}

struct _gpudata {
  CUdeviceptr ptr;
  CUevent ev;
  size_t sz;
  cuda_context *ctx;
  int flags;
  unsigned int refcnt;
};

COMPYTE_LOCAL gpudata *cuda_make_buf(void *c, CUdeviceptr p, size_t sz) {
    cuda_context *ctx = (cuda_context *)c;
    gpudata *res;

    res = malloc(sizeof(*res));
    if (res == NULL) return NULL;
    res->refcnt = 1;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      return NULL;
    }

    res->ptr = p;
    ctx->err = cuEventCreate(&res->ev,
		      CU_EVENT_DISABLE_TIMING|CU_EVENT_BLOCKING_SYNC);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      cuda_exit(ctx);
      return NULL;
    }
    res->sz = sz;
    res->flags = DONTFREE;
    res->ctx = ctx;
    ctx->refcnt++;

    cuEventRecord(res->ev, ctx->s);
    cuda_exit(ctx);
    return res;
}

COMPYTE_LOCAL CUdeviceptr cuda_get_ptr(gpudata *g) { return g->ptr; }
COMPYTE_LOCAL size_t cuda_get_sz(gpudata *g) { return g->sz; }

/* The total size of the arguments is limited to 256 bytes */
#define NUM_ARGS (256/sizeof(void*))

struct _gpukernel {
    CUmodule m;
    CUfunction k;
    CUevent ev;
    void *args[NUM_ARGS];
#if CUDA_VERSION < 4000
    size_t types[NUM_ARGS];
#endif
    unsigned int argcount;
    gpudata *bs[NUM_ARGS];
    cuda_context *ctx;
    unsigned int refcnt;
};

#define FAIL(v, e) { if (ret) *ret = e; return v; }
#define CHKFAIL(v) if (err != CUDA_SUCCESS) FAIL(v, GA_IMPL_ERROR)

static const char CUDA_PREAMBLE[] =
    "#define local_barrier() __syncthreads()\n"
    "#define WITHIN_KERNEL extern \"C\" __device__\n"
    "#define KERNEL extern \"C\" __global__\n"
    "#define GLOBAL_MEM /* empty */\n"
    "#define LOCAL_MEM __shared__\n"
    "#define LOCAL_MEM_ARG /* empty */\n"
    "#define REQD_WG_SIZE(X,Y,Z) __launch_bounds__(X*Y, Z)\n"
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
    "#define GDIM_2 gridDim.z\n"
    "#ifdef _MSC_VER\n"
    "#define signed __int8 int8_t\n"
    "#define unsigned __int8 uint8_t\n"
    "#define signed __int16 int16_t\n"
    "#define unsigned __int16 uint16_t\n"
    "#define signed __int32 int32_t\n"
    "#define unsigned __int32 uint32_t\n"
    "#define signed __int64 int64_t\n"
    "#define unsigned __int64 uint64_t\n"
    "#else\n"
    "#include <stdint.h>\n"
    "#endif\n"
    "#define ga_bool uint8_t\n"
    "#define ga_byte int8_t\n"
    "#define ga_ubyte uint8_t\n"
    "#define ga_short int16_t\n"
    "#define ga_ushort uint16_t\n"
    "#define ga_int int32_t\n"
    "#define ga_uint uint32_t\n"
    "#define ga_long int64_t\n"
    "#define ga_ulong uint64_t\n"
    "#define ga_float float\n"
    "#define ga_double double\n"
    "#define ga_half uint16_t\n";
/* XXX: add complex, quads, longlong */
/* XXX: add vector types */

static int cuda_property(void *, gpudata *, gpukernel *, int, void *);

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
#endif
#if CUDA_VERSION >= 4020
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
    cuda_context *res;
    static int init_done = 0;

    if (ord == -1) {
      /* Grab the ambient context */
      err = cuCtxGetCurrent(&ctx);
      CHKFAIL(NULL);
      res = cuda_make_ctx(ctx, DONTFREE);
      if (res == NULL) {
        FAIL(NULL, GA_IMPL_ERROR);
      }
      return res;
    }

    if (!init_done) {
      err = cuInit(0);
      CHKFAIL(NULL);
      init_done = 1;
    }
    err = cuDeviceGet(&dev, ord);
    CHKFAIL(NULL);
    err = cuCtxCreate(&ctx, CU_CTX_SCHED_YIELD, dev);
    CHKFAIL(NULL);
    res = cuda_make_ctx(ctx, 0);
    if (res == NULL) {
      cuCtxDestroy(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }
    /* Don't leave the context on the thread stack */
    cuCtxPopCurrent(NULL);
    return res;
}

static void cuda_deinit(void *c) {
  cuda_free_ctx((cuda_context *)c);
}

static gpudata *cuda_alloc(void *c, size_t size, int *ret) {
    gpudata *res;
    cuda_context *ctx = (cuda_context *)c;

    res = malloc(sizeof(*res));
    if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
    res->refcnt = 1;

    res->sz = size;
    res->flags = 0;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    ctx->err = cuEventCreate(&res->ev,
                             CU_EVENT_DISABLE_TIMING|CU_EVENT_BLOCKING_SYNC);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    if (size == 0) size = 1;

    ctx->err = cuMemAlloc(&res->ptr, size);
    if (ctx->err != CUDA_SUCCESS) {
        cuEventDestroy(res->ev);
        free(res);
        cuda_exit(ctx);
        FAIL(NULL, GA_IMPL_ERROR);
    }
    res->ctx = ctx;
    ctx->refcnt++;
    cuEventRecord(res->ev, ctx->s);
    cuda_exit(ctx);
    return res;
}

static void cuda_retain(gpudata *d) {
  d->refcnt++;
}

static void cuda_free(gpudata *d) {
  CUresult err;
  /* We ignore errors on free */
  d->refcnt--;
  if (d->refcnt == 0) {
    cuda_enter(d->ctx);
    err = cuEventQuery(d->ev);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "ERROR: in cuda_release: buffer refcount reached 0 "
              "but events are still pending.");
      cuEventSynchronize(d->ev);
    }
    if (!(d->flags & DONTFREE))
      cuMemFree(d->ptr);
    cuEventDestroy(d->ev);
    cuda_exit(d->ctx);
    cuda_free_ctx(d->ctx);
    free(d);
  }
}

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
  return (a->ctx == b->ctx && a->sz != 0 && b->sz != 0 &&
          ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
           (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr)));
}

static int cuda_move(gpudata *dst, size_t dstoff, gpudata *src,
                     size_t srcoff, size_t sz) {
    cuda_context *ctx = dst->ctx;
    int res = GA_NO_ERROR;
    if (src->ctx != dst->ctx) return GA_VALUE_ERROR;

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz || (src->sz -srcoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuStreamWaitEvent(ctx->s, src->ev, 0);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuStreamWaitEvent(ctx->s, dst->ev, 0);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemcpyDtoDAsync(dst->ptr + dstoff, src->ptr + srcoff, sz,
                                 ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuEventRecord(src->ev, ctx->s);
    cuEventRecord(dst->ev, ctx->s);
    cuda_exit(ctx);
    return res;
}

static int cuda_read(void *dst, gpudata *src, size_t srcoff, size_t sz) {
    cuda_context *ctx = src->ctx;

    if (sz == 0) return GA_NO_ERROR;

    if ((src->sz - srcoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuStreamWaitEvent(ctx->s, src->ev, 0);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemcpyDtoHAsync(dst, src->ptr + srcoff, sz, ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuEventRecord(src->ev, ctx->s);
    /* We want the copy to be finished when the function returns */
    cuEventSynchronize(src->ev);
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_write(gpudata *dst, size_t dstoff, const void *src,
                      size_t sz) {
    cuda_context *ctx = dst->ctx;

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuStreamWaitEvent(ctx->s, dst->ev, 0);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemcpyHtoDAsync(dst->ptr + dstoff, src, sz, ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuEventRecord(dst->ev, ctx->s);
    cuEventSynchronize(dst->ev);
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, size_t dstoff, int data) {
    cuda_context *ctx = dst->ctx;
    if ((dst->sz - dstoff) == 0) return GA_NO_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuStreamWaitEvent(ctx->s, dst->ev, 0);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemsetD8Async(dst->ptr + dstoff, data, dst->sz - dstoff,
                               ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuEventRecord(dst->ev, ctx->s);
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static const char *detect_arch(int *ret) {
    CUdevice dev;
    int major, minor;
    CUresult err;
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
#ifdef DEBUG
#define NVCC_ARGS "nvcc", "-g", "-G", "-arch", arch_arg, "-x", "cu", \
      "--cubin", namebuf, "-o", outbuf
#else
#define NVCC_ARGS "nvcc", "-arch", arch_arg, "-x", "cu", \
      "--cubin", namebuf, "-o", outbuf
#endif
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

    buf = malloc((size_t)st.st_size);
    if (buf == NULL) {
        close(fd);
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    s = read(fd, buf, (size_t)st.st_size);
    close(fd);
    unlink(outbuf);
    /* fd is not non-blocking; should have complete read */
    if (s == -1) {
        free(buf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    return buf;
}

static void *(*call_compiler)(const char *src, size_t len, int *ret) = call_compiler_impl;

COMPYTE_LOCAL void cuda_set_compiler(void *(*compiler_f)(const char *, size_t,
                                                         int *)) {
    if (compiler_f == NULL) {
        call_compiler = call_compiler_impl;
    } else {
        call_compiler = compiler_f;
    }
}

static gpukernel *cuda_newkernel(void *c, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, int flags, int *ret) {
    cuda_context *ctx = (cuda_context *)c;
    struct iovec *descr;
    char *buf;
    char *p;
    gpukernel *res;
    size_t tot_len;
    CUdevice dev;
    unsigned int i;
    unsigned int pre;
    int ptx_mode = 0;
    int major, minor;

    if (count == 0) FAIL(NULL, GA_VALUE_ERROR);

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      FAIL(NULL, GA_IMPL_ERROR);

    ctx->err = cuCtxGetDevice(&dev);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }
    ctx->err = cuDeviceComputeCapability(&major, &minor, dev);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    // GA_USE_CLUDA is done later
    // GA_USE_SMALL will always work
    if (flags & GA_USE_DOUBLE) {
      if (major < 1 || (major == 1 && minor < 3)) {
        cuda_exit(ctx);
        FAIL(NULL, GA_DEVSUP_ERROR);
      }
    }
    if (flags & GA_USE_COMPLEX) {
      // just for now since it is most likely broken
      cuda_exit(ctx);
      FAIL(NULL, GA_DEVSUP_ERROR);
    }
    // GA_USE_HALF should always work

    pre = 0;
    if (flags & GA_USE_CLUDA) pre++;
    descr = calloc(count+pre, sizeof(*descr));
    if (descr == NULL) {
      cuda_exit(ctx);
      FAIL(NULL, GA_SYS_ERROR);
    }

    if (flags & GA_USE_PTX) {
        ptx_mode = 1;
    }

    if (flags & GA_USE_CLUDA) {
        descr[0].iov_base = (void *)CUDA_PREAMBLE;
        descr[0].iov_len = strlen(CUDA_PREAMBLE);
    }
    tot_len = descr[0].iov_len;

    if (lengths == NULL) {
        for (i = 0; i < count; i++) {
            descr[i+pre].iov_base = (void *)strings[i];
            descr[i+pre].iov_len = strlen(strings[i]);
            tot_len += descr[i+pre].iov_len;
        }
    } else {
        for (i = 0; i < count; i++) {
            descr[i+pre].iov_base = (void *)strings[i];
            descr[i+pre].iov_len = lengths[i]?lengths[i]:strlen(strings[i]);
            tot_len += descr[i+pre].iov_len;
        }
    }

    if (ptx_mode) tot_len += 1;

    buf = malloc(tot_len);
    if (buf == NULL) {
        free(descr);
        cuda_exit(ctx);
        FAIL(NULL, GA_SYS_ERROR);
    }

    p = buf;
    for (i = 0; i < count+pre; i++) {
        memcpy(p, descr[i].iov_base, descr[i].iov_len);
        p += descr[i].iov_len;
    }

    if (ptx_mode) p[0] = '\0';

    free(descr);

    if (ptx_mode) {
        p = buf;
    } else {
        p = call_compiler(buf, tot_len, ret);
        free(buf);
        if (p == NULL) {
          cuda_exit(ctx);
          return NULL;
        }
    }

    res = malloc(sizeof(*res));
    if (res == NULL) {
      cuda_exit(ctx);
      FAIL(NULL, GA_SYS_ERROR);
    }
    res->refcnt = 1;

    memset(res, 0, sizeof(*res));

    ctx->err = cuModuleLoadData(&res->m, p);
    free(p);

    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    ctx->err = cuModuleGetFunction(&res->k, res->m, fname);
    if (ctx->err != CUDA_SUCCESS) {
        cuModuleUnload(res->m);
        free(res);
        cuda_exit(ctx);
        FAIL(NULL, GA_IMPL_ERROR);
    }

    res->ctx = ctx;
    ctx->refcnt++;
    cuda_exit(ctx);
    return res;
}

static void cuda_retainkernel(gpukernel *k) {
  k->refcnt++;
}

static void cuda_freekernel(gpukernel *k) {
  /* We don't check for errors on free */
  unsigned int i;

  k->refcnt--;
  if (k->refcnt == 0) {
    cuda_enter(k->ctx);
    for (i = 0; i < k->argcount; i++) {
      free(k->args[i]);
      if (k->bs[i] != NULL)
        cuda_free(k->bs[i]);
    }
    cuModuleUnload(k->m);
    cuda_exit(k->ctx);
    cuda_free_ctx(k->ctx);
    free(k);
  }
}

static int cuda_setkernelarg(gpukernel *k, unsigned int index, int typecode,
                             const void *val) {
    void *tmp;
    gpudata *b;
    size_t sz;
    if (index >= NUM_ARGS) return GA_VALUE_ERROR;

    if (index >= k->argcount)
        k->argcount = index+1;

    if (typecode == GA_BUFFER) {
        b = (gpudata *)val;
        if (k->ctx != b->ctx)
            return GA_VALUE_ERROR;
        if (k->bs[index] != NULL)
          cuda_free(k->bs[index]);
        k->bs[index] = b;
        cuda_retain(k->bs[index]);
        sz = sizeof(CUdeviceptr);
        val = &b->ptr;
    } else {
        sz = compyte_get_elsize(typecode);
        if (k->bs[index] != NULL)
          cuda_free(k->bs[index]);
        k->bs[index] = NULL;
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

#define ALIGN_UP(offset, align) ((offset) + (align) - 1) & ~((align) - 1)

static int cuda_callkernel(gpukernel *k, size_t bs, size_t gs) {
    cuda_context *ctx = k->ctx;
    unsigned int i;
#if CUDA_VERSION < 4000
    size_t total = 0;
    size_t align, sz;
#endif

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    for (i = 0; i < k->argcount; i++) {
        if (k->bs[i] != NULL) {
            ctx->err = cuStreamWaitEvent(ctx->s, k->bs[i]->ev, 0);
            if (ctx->err != CUDA_SUCCESS) {
              cuda_exit(ctx);
              return GA_IMPL_ERROR;
            }
        }
    }

#if CUDA_VERSION >= 4000
    ctx->err = cuLaunchKernel(k->k, gs, 1, 1, bs, 1, 1, 0, ctx->s, k->args,
                              NULL);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
#else
    for (i = 0; i < k->argcount; i++) {
        if (k->types[i] == GA_DELIM) {
            align = DEVPTR_ALIGN;
            sz = sizeof(CUdeviceptr);
        } else {
            align = compyte_get_type(k->types[i])->align;
            sz = compyte_get_elsize(k->types[i]);
        }
        total = ALIGN_UP(total, align);
        ctx->err = cuParamSetv(k->k, (int)total, k->args[i], (unsigned int)sz);
        if (ctx->err != CUDA_SUCCESS) {
          cuda_exit(ctx);
          return GA_IMPL_ERROR;
        }
        total += sz;
    }
    ctx->err = cuParamSetSize(k->k, total);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuFuncSetBlockShape(k->k, bs, 1, 1);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuLaunchGridAsync(k->k, gs, 1, ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
#endif
    for (i = 0; i < k->argcount; i++) {
      if (k->bs[i] != NULL)
        cuEventRecord(k->bs[i]->ev, ctx->s);
    }
    cuEventRecord(k->ev, ctx->s);
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_sync(gpudata *b) {
  cuda_context *ctx = (cuda_context *)b->ctx;

  cuda_enter(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;
  ctx->err = cuEventSynchronize(b->ev);
  cuda_exit(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;
  return GA_NO_ERROR;
}

static const char ELEM_HEADER_PTX[] = ".version 3.0\n.target %s\n\n"
    ".entry extcpy (\n"
    ".param .u%u a_data,\n"
    ".param .u%u b_data ) {\n"
    ".reg .u16 rh1, rh2;\n"
    ".reg .u32 r1;\n"
    ".reg .u%u numThreads, i, a_pi, b_pi, a_p, b_p, rl1;\n"
    ".reg .u%u rp1, rp2;\n"
    ".reg .%s tmpa;\n"
    ".reg .%s tmpb;\n"
    ".reg .pred p;\n"
    "mov.u16 rh1, %%ntid.x;\n"
    "mov.u16 rh2, %%ctaid.x;\n"
    "mul.wide.u16 r1, rh1, rh2;\n"
    "cvt.u%u.u32 i, r1;\n"
    "mov.u32 r1, %%tid.x;\n"
    "cvt.u%u.u32 rl1, r1;\n"
    "add.u%u i, i, rl1;\n"
    "mov.u16 rh2, %%nctaid.x;\n"
    "mul.wide.u16 r1, rh2, rh1;\n"
    "cvt.u%u.u32 numThreads, r1;\n"
    "setp.ge.u%u p, i, %" SPREFIX "uU;\n"
    "@p bra $end;\n"
    "$loop_begin:\n"
    "mov.u%u a_p, 0U;\n"
    "mov.u%u b_p, 0U;\n";

static inline ssize_t ssabs(ssize_t v) {
    return (v < 0 ? -v : v);
}

static int cuda_perdim_ptx(char *strs[], unsigned int *count, unsigned int nd,
                           const size_t *dims, const ssize_t *str,
                           const char *id, unsigned int bits) {
    int i;

    if (nd > 0) {
        if (asprintf(&strs[*count], "mov.u%u %si, i;\n", bits, id) == -1)
            return -1;
        (*count)++;

        for (i = nd-1; i > 0; i--) {
            if (asprintf(&strs[*count], "rem.u%u rl1, %si, %" SPREFIX "uU;\n"
                         "mad.lo.s%u %s, rl1, %" SPREFIX "d, %s;\n"
                         "div.u%u %si, %si, %" SPREFIX "uU;\n",
                         bits, id, dims[i],
                         bits, id, str[i], id,
                         bits, id, id, dims[i]) == -1)
                return -1;
            (*count)++;
        }

        if (asprintf(&strs[*count], "mad.lo.s%u %s, %si, %" SPREFIX "d, %s;\n",
                     bits, id, id, str[0], id) == -1)
            return -1;
        (*count)++;
    }
    return 0;
}

static const char ELEM_FOOTER_PTX[] = "add.u%u i, i, numThreads;\n"
    "setp.lt.u%u p, i, %" SPREFIX "uU;\n"
    "@p bra $loop_begin;\n"
    "$end:\n"
    "ret;\n"
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

static inline const char *get_rmod(int intype, int outtype) {
    switch (intype) {
    case GA_DOUBLE:
        if (outtype == GA_HALF || outtype == GA_FLOAT) return ".rn";
    case GA_FLOAT:
        if (outtype == GA_HALF) return ".rn";
    case GA_HALF:
        switch (outtype) {
        case GA_BYTE:
        case GA_UBYTE:
        case GA_BOOL:
        case GA_SHORT:
        case GA_USHORT:
        case GA_INT:
        case GA_UINT:
        case GA_LONG:
        case GA_ULONG:
            return ".rni";
        }
        break;
    case GA_BYTE:
    case GA_UBYTE:
    case GA_BOOL:
    case GA_SHORT:
    case GA_USHORT:
    case GA_INT:
    case GA_UINT:
    case GA_LONG:
    case GA_ULONG:
        switch (outtype) {
        case GA_HALF:
        case GA_FLOAT:
        case GA_DOUBLE:
            return ".rn";
        }
    }
    return "";
}

static inline unsigned int xmin(unsigned long a, unsigned long b) {
    return (unsigned int)((a < b) ? a : b);
}

static int cuda_extcopy(gpudata *input, size_t ioff, gpudata *output, size_t ooff,
                        int intype, int outtype, unsigned int a_nd,
                        const size_t *a_dims, const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str) {
    char *strs[64];
    unsigned int count = 0;
    int res = GA_SYS_ERROR;
    
    size_t nEls = 1, ls, gs;
    gpukernel *k;
    unsigned int i;
    int flags = GA_USE_PTX;

    unsigned int bits = sizeof(void *)*8;
    const char *in_t;
    const char *out_t;
    const char *rmod;
    const char *arch;

    if (input->ctx != output->ctx)
      return GA_INVALID_ERROR;

    for (i = 0; i < a_nd; i++) {
        nEls *= a_dims[i];
    }
    if (nEls == 0) return GA_NO_ERROR;

    in_t = map_t(intype);
    out_t = map_t(outtype);
    rmod = get_rmod(intype, outtype);
    if (in_t == NULL || out_t == NULL) return GA_DEVSUP_ERROR;
    arch = detect_arch(&res);
    if (arch == NULL) return res;

    if (asprintf(&strs[count], ELEM_HEADER_PTX, arch, bits, bits, bits,
		 bits, in_t, out_t, bits, bits, bits, bits, bits, nEls,
		 bits, bits) == -1)
        goto fail;
    count++;

    if (cuda_perdim_ptx(strs, &count, a_nd, a_dims, a_str, "a_p", bits) == -1)
        goto fail;
    if (cuda_perdim_ptx(strs, &count, b_nd, b_dims, b_str, "b_p", bits) == -1)
        goto fail;

    if (asprintf(&strs[count], "ld.param.u%u rp1, [a_data];\n"
                 "cvt.s%u.s%u rp2, a_p;\n"
                 "add.s%u rp1, rp1, rp2;\n"
                 "ld.global.%s tmpa, [rp1+%" SPREFIX "u];\n"
                 "cvt%s.%s.%s tmpb, tmpa;\n"
                 "ld.param.u%u rp1, [b_data];\n"
                 "cvt.s%u.s%u rp2, b_p;\n"
                 "add.s%u rp1, rp1, rp2;\n"
                 "st.global.%s [rp1+%" SPREFIX "u], tmpb;\n", bits,
                 bits, bits,
                 bits,
		 in_t, ioff,
                 rmod, out_t, in_t,
                 bits,
                 bits, bits,
                 bits,
                 out_t, ooff) == -1)
        goto fail;
    count++;

    if (asprintf(&strs[count], ELEM_FOOTER_PTX, bits, bits, nEls) == -1)
        goto fail;
    count++;

    assert(count < (sizeof(strs)/sizeof(strs[0])));

    if (intype == GA_DOUBLE || outtype == GA_DOUBLE ||
        intype == GA_CDOUBLE || outtype == GA_CDOUBLE) {
        flags |= GA_USE_DOUBLE;
    }

    if (outtype == GA_HALF || intype == GA_HALF) {
        flags |= GA_USE_HALF;
    }

    if (compyte_get_elsize(outtype) < 4 || compyte_get_elsize(intype) < 4) {
        /* Should check for non-mod4 strides too */
        flags |= GA_USE_SMALL;
    }

    if (outtype == GA_CFLOAT || intype == GA_CFLOAT ||
        outtype == GA_CDOUBLE || intype == GA_CDOUBLE) {
        flags |= GA_USE_COMPLEX;
    }

    k = cuda_newkernel(input->ctx, count, (const char **)strs, NULL, "extcpy",
                       flags, &res);
    if (k == NULL) goto fail;
    res = cuda_setkernelarg(k, 0, GA_BUFFER, input);
    if (res != GA_NO_ERROR) goto failk;
    res = cuda_setkernelarg(k, 1, GA_BUFFER, output);
    if (res != GA_NO_ERROR) goto failk;

    /* Cheap kernel scheduling */
    res = cuda_property(NULL, NULL, k, GA_KERNEL_PROP_MAXLSIZE, &ls);
    if (res != GA_NO_ERROR) goto failk;

    gs = ((nEls-1) / ls) + 1;
    res = cuda_callkernel(k, ls, gs);

failk:
    cuda_freekernel(k);
fail:
    for (i = 0; i < count; i++) {
        free(strs[i]);
    }
    return res;
}

static int cuda_property(void *c, gpudata *buf, gpukernel *k, int prop_id,
                         void *res) {
  cuda_context *ctx = NULL;
  if (c != NULL) {
    ctx = (cuda_context *)c;
  } else if (buf != NULL) {
    ctx = buf->ctx;
  } else if (k != NULL) {
    ctx = k->ctx;
  }
  if (ctx == NULL) {
    return GA_VALUE_ERROR;
  }
  /* I know that 512 and 1024 are magic numbers.
     There is an indication in buffer.h, though. */
  if (prop_id < 512) {
    if (c == NULL)
      return GA_VALUE_ERROR;
  } else if (prop_id < 1024) {
    if (buf == NULL)
      return GA_VALUE_ERROR;
  } else {
    if (k == NULL)
      return GA_VALUE_ERROR;
  }

  cuda_enter(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;

  switch (prop_id) {
    char *s;
    CUdevice id;
    int i;
  case GA_CTX_PROP_DEVNAME:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    /* 256 is what the CUDA API uses so it's good enough for me */
    s = malloc(256);
    if (s == NULL) {
      cuda_exit(ctx);
      return GA_MEMORY_ERROR;
    }
    ctx->err = cuDeviceGetName(s, 256, id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((char **)res) = s;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  case GA_CTX_PROP_MAXLSIZE:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                    id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((size_t *)res) = i;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  case GA_CTX_PROP_LMEMSIZE:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                                    id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((size_t *)res) = i;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  case GA_CTX_PROP_NUMPROCS:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuDeviceGetAttribute(&i,
                                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((unsigned int *)res) = i;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  case GA_CTX_PROP_MAXGSIZE:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                    id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((size_t *)res) = i;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  case GA_BUFFER_PROP_REFCNT:
    *((unsigned int *)res) = buf->refcnt;
    return GA_NO_ERROR;
  case GA_BUFFER_PROP_CTX:
  case GA_KERNEL_PROP_CTX:
    *((void **)res) = (void *)ctx;
    return GA_NO_ERROR;
  case GA_KERNEL_PROP_MAXLSIZE:
    ctx->err = cuFuncGetAttribute(&i,
                                  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  k->k);
    cuda_exit(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
    *((size_t *)res) = i;
    return GA_NO_ERROR;
  case GA_KERNEL_PROP_PREFLSIZE:
    ctx->err = cuCtxGetDevice(&id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    ctx->err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_WARP_SIZE, id);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    *((size_t *)res) = i;
    cuda_exit(ctx);
    return GA_NO_ERROR;
  default:
    cuda_exit(ctx);
    return GA_INVALID_ERROR;
  }
}

static const char *cuda_error(void *c) {
  cuda_context *ctx = (cuda_context *)c;
  if (ctx == NULL)
    return get_error_string(err);
  else
    return get_error_string(ctx->err);
}

COMPYTE_LOCAL
const compyte_buffer_ops cuda_ops = {cuda_init,
                                     cuda_deinit,
                                     cuda_alloc,
                                     cuda_retain,
                                     cuda_free,
                                     cuda_share,
                                     cuda_move,
                                     cuda_read,
                                     cuda_write,
                                     cuda_memset,
                                     cuda_newkernel,
                                     cuda_retainkernel,
                                     cuda_freekernel,
                                     cuda_setkernelarg,
                                     cuda_callkernel,
                                     cuda_sync,
                                     cuda_extcopy,
                                     cuda_property,
                                     cuda_error};
