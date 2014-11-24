#define _CRT_SECURE_NO_WARNINGS

#include "private.h"
#include "private_cuda.h"

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

#include "util/strb.h"

#include "gpuarray/buffer.h"
#include "gpuarray/util.h"
#include "gpuarray/error.h"
#include "gpuarray/extension.h"
#include "gpuarray/buffer_blas.h"

typedef struct {char c; CUdeviceptr x; } st_devptr;
#define DEVPTR_ALIGN (sizeof(st_devptr) - sizeof(CUdeviceptr))

static CUresult err;

static void cuda_free(gpudata *);
static void cuda_freekernel(gpukernel *);
static int cuda_property(void *, gpudata *, gpukernel *, int, void *);

#define val_free(v) cuda_freekernel(*v);
#include "cache_extcopy.h"

static int detect_arch(char *ret);

void *cuda_make_ctx(CUcontext ctx, int flags) {
  cuda_context *res;
  res = malloc(sizeof(*res));
  if (res == NULL)
    return NULL;
  res->ctx = ctx;
  res->err = CUDA_SUCCESS;
  res->blas_handle = NULL;
  res->refcnt = 1;
  res->flags = flags;
  if (detect_arch(res->bin_id)) {
    free(res);
    return NULL;
  }
  res->extcopy_cache = cache_alloc(64, 32);
  if (res->extcopy_cache == NULL) {
    free(res);
    return NULL;
  }
  err = cuStreamCreate(&res->s, 0);
  if (err != CUDA_SUCCESS) {
    cache_free(res->extcopy_cache);
    free(res);
    return NULL;
  }
  TAG_CTX(res);
  return res;
}

static void cuda_free_ctx(cuda_context *ctx) {
  gpuarray_blas_ops *blas_ops;

  ASSERT_CTX(ctx);
  ctx->refcnt--;
  if (ctx->refcnt == 0) {
    if (ctx->blas_handle != NULL) {
      ctx->err = cuda_property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas_ops);
      blas_ops->teardown(ctx);
    }
    cuStreamDestroy(ctx->s);
    if (!(ctx->flags & DONTFREE))
      cuCtxDestroy(ctx->ctx);
    cache_free(ctx->extcopy_cache);
    CLEAR(ctx);
    free(ctx);
  }
}

CUcontext cuda_get_ctx(void *ctx) {
  ASSERT_CTX((cuda_context *)ctx);
  return ((cuda_context *)ctx)->ctx;
}

CUstream cuda_get_stream(void *ctx) {
  ASSERT_CTX((cuda_context *)ctx);
  return ((cuda_context *)ctx)->s;
}

void cuda_enter(cuda_context *ctx) {
  ASSERT_CTX(ctx);
  cuCtxGetCurrent(&ctx->old);
  if (ctx->old != ctx->ctx)
    ctx->err = cuCtxSetCurrent(ctx->ctx);
  /* If no context was there in the first place, then we take over
     to avoid the set dance on the thread */
  if (ctx->old == NULL) ctx->old = ctx->ctx;
}

void cuda_exit(cuda_context *ctx) {
  if (ctx->old != ctx->ctx)
    cuCtxSetCurrent(ctx->old);
}

gpudata *cuda_make_buf(void *c, CUdeviceptr p, size_t sz) {
    cuda_context *ctx = (cuda_context *)c;
    gpudata *res;
    int flags = CU_EVENT_DISABLE_TIMING;

    res = malloc(sizeof(*res));
    if (res == NULL) return NULL;
    res->refcnt = 1;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      return NULL;
    }

    res->ptr = p;
    if (ctx->flags & GA_CTX_MULTI_THREAD)
      flags |= CU_EVENT_BLOCKING_SYNC;
    ctx->err = cuEventCreate(&res->ev, flags);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      cuda_exit(ctx);
      return NULL;
    }
    res->sz = sz;
    res->flags = DONTFREE;
    res->ctx = ctx;
    ctx->refcnt++;

    cuda_exit(ctx);
    TAG_BUF(res);
    return res;
}

CUdeviceptr cuda_get_ptr(gpudata *g) { ASSERT_BUF(g); return g->ptr; }
size_t cuda_get_sz(gpudata *g) { ASSERT_BUF(g); return g->sz; }

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
    "#define ga_half uint16_t\n"
    "#define ga_size size_t\n";
/* XXX: add complex, quads, longlong */
/* XXX: add vector types */

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
    case CUDA_ERROR_PROFILER_DISABLED: return "Profiler is disabled";
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED: return "Profiler is not initialized";
    case CUDA_ERROR_PROFILER_ALREADY_STARTED: return "Profiler has already started";
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED: return "Profiler has already stopped";
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE: return "Context is already bound to another thread";
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "Peer access already enabled";
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: return "Peer access not enabled";
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "Primary context already initialized";
    case CUDA_ERROR_CONTEXT_IS_DESTROYED: return "Context has been destroyed (or not yet initialized)";
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

static void *cuda_init(int ord, int flags, int *ret) {
    CUdevice dev;
    CUcontext ctx;
    cuda_context *res;
    static int init_done = 0;
    unsigned int fl = CU_CTX_SCHED_AUTO;

    if (ord == -1) {
      /* Grab the ambient context */
      err = cuCtxGetCurrent(&ctx);
      CHKFAIL(NULL);
      res = cuda_make_ctx(ctx, DONTFREE);
      if (res == NULL) {
        FAIL(NULL, GA_IMPL_ERROR);
      }
      res->flags |= flags;
      return res;
    }

    if (!init_done) {
      err = cuInit(0);
      CHKFAIL(NULL);
      init_done = 1;
    }
    err = cuDeviceGet(&dev, ord);
    CHKFAIL(NULL);
    if (flags & GA_CTX_SINGLE_THREAD)
      fl = CU_CTX_SCHED_SPIN;
    if (flags & GA_CTX_MULTI_THREAD)
      fl = CU_CTX_SCHED_YIELD;
    err = cuCtxCreate(&ctx, fl, dev);
    CHKFAIL(NULL);
    res = cuda_make_ctx(ctx, 0);
    res->flags |= flags;
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

static gpudata *cuda_alloc(void *c, size_t size, void *data, int flags,
			   int *ret) {
    gpudata *res;
    cuda_context *ctx = (cuda_context *)c;
    int fl = CU_EVENT_DISABLE_TIMING;

    if ((flags & GA_BUFFER_INIT) && data == NULL) FAIL(NULL, GA_VALUE_ERROR);
    if ((flags & (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) ==
	(GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) FAIL(NULL, GA_VALUE_ERROR);

    /* TODO: figure out how to make this work */
    if (flags & GA_BUFFER_HOST) FAIL(NULL, GA_DEVSUP_ERROR);

    res = malloc(sizeof(*res));
    if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
    res->refcnt = 1;

    res->sz = size;
    res->flags = flags & (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY);

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS) {
      free(res);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    if (ctx->flags & GA_CTX_MULTI_THREAD)
      fl |= CU_EVENT_BLOCKING_SYNC;
    ctx->err = cuEventCreate(&res->ev, fl);

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

    if (flags & GA_BUFFER_INIT) {
      ctx->err = cuMemcpyHtoD(res->ptr, data, size);
      if (ctx->err != CUDA_SUCCESS) {
	cuda_free(res);
	FAIL(NULL, GA_IMPL_ERROR)
      }
    }

    cuda_exit(ctx);
    TAG_BUF(res);
    return res;
}

static void cuda_retain(gpudata *d) {
  ASSERT_BUF(d);
  d->refcnt++;
}

static void cuda_free(gpudata *d) {
  /* We ignore errors on free */
  ASSERT_BUF(d);
  d->refcnt--;
  if (d->refcnt == 0) {
    cuda_enter(d->ctx);
    /*
     * From testing, I have discovered that cuMemFree() will just
     * block until nothing uses the region on the GPU.  Since this is
     * not documented behavior, we will emulate that here.
     */
    cuEventSynchronize(d->ev);
    if (!(d->flags & DONTFREE))
      cuMemFree(d->ptr);
    cuEventDestroy(d->ev);
    cuda_exit(d->ctx);
    cuda_free_ctx(d->ctx);
    CLEAR(d);
    free(d);
  }
}

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
  ASSERT_BUF(a);
  ASSERT_BUF(b);
  return (a->ctx == b->ctx && a->sz != 0 && b->sz != 0 &&
          ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
           (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr)));
}

static int cuda_move(gpudata *dst, size_t dstoff, gpudata *src,
                     size_t srcoff, size_t sz) {
    cuda_context *ctx = dst->ctx;
    int res = GA_NO_ERROR;
    ASSERT_BUF(dst);
    ASSERT_BUF(src);
    if (src->ctx != dst->ctx) return GA_VALUE_ERROR;

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz || (src->sz - srcoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuMemcpyDtoDAsync(dst->ptr + dstoff, src->ptr + srcoff, sz,
                                 ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuda_exit(ctx);
    return res;
}

static int cuda_read(void *dst, gpudata *src, size_t srcoff, size_t sz) {
    cuda_context *ctx = src->ctx;

    ASSERT_BUF(src);

    if (sz == 0) return GA_NO_ERROR;

    if ((src->sz - srcoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuEventSynchronize(src->ev);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemcpyDtoH(dst, src->ptr + srcoff, sz);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_write(gpudata *dst, size_t dstoff, const void *src,
                      size_t sz) {
    cuda_context *ctx = dst->ctx;

    ASSERT_BUF(dst);

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuEventSynchronize(dst->ev);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    ctx->err = cuMemcpyHtoD(dst->ptr + dstoff, src, sz);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, size_t dstoff, int data) {
    cuda_context *ctx = dst->ctx;

    ASSERT_BUF(dst);

    if ((dst->sz - dstoff) == 0) return GA_NO_ERROR;

    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    ctx->err = cuMemsetD8Async(dst->ptr + dstoff, data, dst->sz - dstoff,
                               ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static CUresult get_cc(CUdevice dev, int *maj, int *min) {
#if CUDA_VERSION < 6500
  return cuDeviceComputeCapability(maj, min, dev);
#else
  CUresult lerr;
  lerr = cuDeviceGetAttribute(maj,
                              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                              dev);
  if (lerr != CUDA_SUCCESS)
    return lerr;
  return cuDeviceGetAttribute(min,
                              CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                              dev);
#endif
}

static int detect_arch(char *ret) {
    CUdevice dev;
    int major, minor;
    int res;
    CUresult err;
    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    err = get_cc(dev, &major, &minor);
    if (err != CUDA_SUCCESS) return GA_IMPL_ERROR;
    res = snprintf(ret, 6, "sm_%d%d", major, minor);
    if (res == -1 || res > 6) return GA_UNSUPPORTED_ERROR;
    return GA_NO_ERROR;
}

static const char *TMP_VAR_NAMES[] = {"GPUARRAY_TMPDIR", "TMPDIR", "TMP",
                                      "TEMP", "USERPROFILE"};

static void *call_compiler_impl(const char *src, size_t len, size_t *bin_len,
                                int *ret) {
    char namebuf[PATH_MAX];
    char outbuf[PATH_MAX];
    char *tmpdir;
    char arch_arg[6]; /* Must be at least 6, see detect_arch() */
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

    res = detect_arch(arch_arg);
    if (res != GA_NO_ERROR) FAIL(NULL, res);

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
    strlcat(namebuf, "/gpuarray.cuda.XXXXXXXX", sizeof(namebuf));

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
#define NVCC_ARGS NVCC_BIN, "-g", "-G", "-arch", arch_arg, "-x", "cu", \
      "--cubin", namebuf, "-o", outbuf
#else
#define NVCC_ARGS NVCC_BIN, "-arch", arch_arg, "-x", "cu", \
      "--cubin", namebuf, "-o", outbuf
#endif
#ifdef _WIN32
    sys_err = _spawnl(_P_WAIT, NVCC_BIN, NVCC_ARGS, NULL);
    unlink(namebuf);
    if (sys_err == -1) FAIL(NULL, GA_SYS_ERROR);
    if (sys_err != 0) FAIL(NULL, GA_RUN_ERROR);
#else
    p = fork();
    if (p == 0) {
        execl(NVCC_BIN, NVCC_ARGS, NULL);
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

    buf = h_malloc((size_t)st.st_size);
    if (buf == NULL) {
        close(fd);
        unlink(outbuf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    s = read(fd, buf, (size_t)st.st_size);
    close(fd);
    unlink(outbuf);
    /* fd is blocking; should have complete read */
    if (s == -1) {
        h_free(buf);
        FAIL(NULL, GA_SYS_ERROR);
    }

    *bin_len = (size_t)st.st_size;
    return buf;
}

static void *(*call_compiler)(const char *src, size_t len, size_t *bin_len, int *ret) = call_compiler_impl;

GPUARRAY_LOCAL void cuda_set_compiler(void *(*compiler_f)(const char *, size_t,
                                                          size_t *, int *)) {
  return;
  /* Disable custom compilers
  if (compiler_f == NULL) {
    call_compiler = call_compiler_impl;
  } else {
    call_compiler = compiler_f;
  }
  */
}

static gpukernel *cuda_newkernel(void *c, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, unsigned int argcount,
                                 const int *types, int flags, int *ret, char **err_str) {
    cuda_context *ctx = (cuda_context *)c;
    strb sb = STRB_STATIC_INIT;
    char *bin;
    gpukernel *res;
    size_t bin_len = 0;
    CUdevice dev;
    unsigned int i;
    int ptx_mode = 0;
    int binary_mode = 0;
    int major, minor;

    if (count == 0) FAIL(NULL, GA_VALUE_ERROR);

    if (flags & GA_USE_OPENCL)
      FAIL(NULL, GA_DEVSUP_ERROR);

    if (flags & GA_USE_BINARY) {
      // GA_USE_BINARY is exclusive
      if (flags & ~GA_USE_BINARY)
        FAIL(NULL, GA_INVALID_ERROR);
      // We need the length for binary data and there is only one blob.
      if (count != 1 || lengths == NULL || lengths[0] == 0)
        FAIL(NULL, GA_VALUE_ERROR);
    }

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

    if (flags & GA_USE_PTX) {
      ptx_mode = 1;
    } else if (flags & GA_USE_BINARY) {
      binary_mode = 1;
    }

    if (binary_mode) {
      bin = h_memdup(strings[0], lengths[0]);
      bin_len = lengths[0];
      if (bin == NULL) {
        cuda_exit(ctx);
        FAIL(NULL, GA_MEMORY_ERROR);
      }
    } else {
      if (flags & GA_USE_CLUDA) {
        strb_appends(&sb, CUDA_PREAMBLE);
      }

      if (lengths == NULL) {
        for (i = 0; i < count; i++)
        strb_appends(&sb, strings[i]);
      } else {
        for (i = 0; i < count; i++) {
          if (lengths[i] == 0)
            strb_appends(&sb, strings[i]);
          else
            strb_appendn(&sb, strings[i], lengths[i]);
        }
      }

      if (ptx_mode) strb_append0(&sb);

      if (strb_error(&sb)) {
        strb_clear(&sb);
        cuda_exit(ctx);
        return NULL;
      }

      if (ptx_mode) {
        bin = sb.s;
      } else {
        bin = call_compiler(sb.s, sb.l, &bin_len, ret);
        if (bin == NULL) {
          if(err_str != NULL) {
            strb debug_msg = STRB_STATIC_INIT;

            // We're substituting debug_msg for a string with this first line:
            strb_appends(&debug_msg, "CUDA kernel build failure ::\n"); 

            gpukernel_source_with_line_numbers(1, (const char **)&sb.s, &sb.l, &debug_msg);
            strb_append0(&debug_msg); // Make sure a final '\0' is present

            if(!strb_error(&debug_msg)) { // Make sure the strb is in a valid state
              *err_str = strndup(debug_msg.s, debug_msg.l);
              if(*err_str == NULL) {
                strb_clear(&sb);
                cuda_exit(ctx);
                return NULL;
              }
            }
          }
          strb_clear(&sb);
          cuda_exit(ctx);
          return NULL;
        }
        strb_clear(&sb);
      }
    }

    res = h_calloc(1, sizeof(*res));
    if (res == NULL) {
      h_free(bin);
      cuda_exit(ctx);
      FAIL(NULL, GA_SYS_ERROR);
    }

    res->bin_sz = bin_len;
    res->bin = bin;
    hattach(res->bin, res);

    res->refcnt = 1;
    res->argcount = argcount;
    res->types = h_calloc(argcount, sizeof(int));
    hattach(res->types, res);
    if (res->types == NULL) {
      h_free(res);
      FAIL(NULL, GA_MEMORY_ERROR);
    }
    memcpy(res->types, types, argcount*sizeof(int));
    res->args = h_calloc(argcount, sizeof(void *));
    hattach(res->args, res);
    if (res->args == NULL) {
      h_free(res);
      FAIL(NULL, GA_MEMORY_ERROR);
    }

    ctx->err = cuModuleLoadData(&res->m, bin);

    if (ctx->err != CUDA_SUCCESS) {
      h_free(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    ctx->err = cuModuleGetFunction(&res->k, res->m, fname);
    if (ctx->err != CUDA_SUCCESS) {
        cuModuleUnload(res->m);
        h_free(res);
        cuda_exit(ctx);
        FAIL(NULL, GA_IMPL_ERROR);
    }

    res->ctx = ctx;
    ctx->refcnt++;
    cuda_exit(ctx);
    TAG_KER(res);
    return res;
}

static void cuda_retainkernel(gpukernel *k) {
  ASSERT_KER(k);
  k->refcnt++;
}

static void cuda_freekernel(gpukernel *k) {
  ASSERT_KER(k);
  k->refcnt--;
  if (k->refcnt == 0) {
    cuda_enter(k->ctx);
    cuModuleUnload(k->m);
    cuda_exit(k->ctx);
    cuda_free_ctx(k->ctx);
    CLEAR(k);
    h_free(k);
  }
}

static int cuda_callkernel(gpukernel *k, size_t bs[2], size_t gs[2],
                           void **args) {
    cuda_context *ctx = k->ctx;
    unsigned int i;

    ASSERT_KER(k);
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;

    for (i = 0; i < k->argcount; i++) {
      if (k->types[i] == GA_BUFFER) {
        k->args[i] = &((gpudata *)args[i])->ptr;
      } else {
        k->args[i] = args[i];
      }
    }

    ctx->err = cuLaunchKernel(k->k, gs[0], gs[1], 1, bs[0], bs[1], 1, 0,
			      ctx->s, k->args, NULL);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_kernelbin(gpukernel *k, size_t *sz, void **obj) {
  void *res = malloc(k->bin_sz);
  if (res == NULL)
    return GA_MEMORY_ERROR;
  memcpy(res, k->bin, k->bin_sz);
  *sz = k->bin_sz;
  *obj = res;
  return GA_NO_ERROR;
}

static int cuda_sync(gpudata *b) {
  cuda_context *ctx = (cuda_context *)b->ctx;

  ASSERT_BUF(b);
  cuda_enter(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;
  ctx->err = cuEventSynchronize(b->ev);
  cuda_exit(ctx);
  if (ctx->err != CUDA_SUCCESS)
    return GA_IMPL_ERROR;
  return GA_NO_ERROR;
}

static const char ELEM_HEADER_PTX[] = ".version 4.0\n.target %s\n\n"
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

static void cuda_perdim_ptx(strb *sb, unsigned int nd,
			    const size_t *dims, const ssize_t *str,
			    const char *id, unsigned int bits) {
  int i;

  if (nd > 0) {
    strb_appendf(sb, "mov.u%u %si, i;\n", bits, id);
    for (i = nd-1; i > 0; i--) {
      strb_appendf(sb, "rem.u%u rl1, %si, %" SPREFIX "uU;\n"
		   "mad.lo.s%u %s, rl1, %" SPREFIX "d, %s;\n"
		   "div.u%u %si, %si, %" SPREFIX "uU;\n",
		   bits, id, dims[i],
		   bits, id, str[i], id,
		   bits, id, id, dims[i]);
    }

    strb_appendf(sb, "mad.lo.s%u %s, %si, %" SPREFIX "d, %s;\n",
		 bits, id, id, str[0], id);
  }
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

static inline int gen_extcopy_kernel(const cache_key_t *a,
				     cuda_context *ctx, cache_val_t *v,
				     size_t nEls) {
  strb sb = STRB_STATIC_INIT;
  int res = GA_SYS_ERROR;
  int flags = GA_USE_PTX;
  unsigned int bits = sizeof(void *)*8;
  int types[2];
  const char *in_t;
  const char *out_t;
  const char *rmod;
  char arch[6]; /* Must be at least 6, see detect_arch() */

  in_t = map_t(a->itype);
  out_t = map_t(a->otype);
  rmod = get_rmod(a->itype, a->otype);
  if (in_t == NULL || out_t == NULL) return GA_DEVSUP_ERROR;
  res = detect_arch(arch);
  if (res != GA_NO_ERROR) return res;

  strb_appendf(&sb, ELEM_HEADER_PTX, arch, bits, bits, bits,
	       bits, in_t, out_t, bits, bits, bits, bits, bits, nEls,
	       bits, bits);

  cuda_perdim_ptx(&sb, a->ind, a->idims, a->istr, "a_p", bits);
  cuda_perdim_ptx(&sb, a->ond, a->odims, a->ostr, "b_p", bits);

  strb_appendf(&sb, "ld.param.u%u rp1, [a_data];\n"
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
	       in_t, a->ioff,
	       rmod, out_t, in_t,
	       bits,
	       bits, bits,
	       bits,
	       out_t, a->ooff);

  strb_appendf(&sb, ELEM_FOOTER_PTX, bits, bits, nEls);

  if (strb_error(&sb))
    goto fail;

  if (a->itype == GA_DOUBLE || a->otype == GA_DOUBLE ||
      a->itype == GA_CDOUBLE || a->otype == GA_CDOUBLE) {
    flags |= GA_USE_DOUBLE;
  }

  if (a->otype == GA_HALF || a->itype == GA_HALF) {
    flags |= GA_USE_HALF;
  }

  if (gpuarray_get_elsize(a->otype) < 4 || gpuarray_get_elsize(a->itype) < 4) {
    /* Should check for non-mod4 strides too */
    flags |= GA_USE_SMALL;
  }

  if (a->otype == GA_CFLOAT || a->itype == GA_CFLOAT ||
      a->otype == GA_CDOUBLE || a->itype == GA_CDOUBLE) {
    flags |= GA_USE_COMPLEX;
  }

  types[0] = types[1] = GA_BUFFER;
  res = GA_NO_ERROR;
  *v = cuda_newkernel(ctx, 1, (const char **)&sb.s, &sb.l, "extcpy",
                      2, types, flags, &res, NULL);
 fail:
  strb_clear(&sb);
  return res;
}

#include <time.h>

static int cuda_extcopy(gpudata *input, size_t ioff, gpudata *output, size_t ooff,
                        int intype, int outtype, unsigned int a_nd,
                        const size_t *a_dims, const ssize_t *a_str,
                        unsigned int b_nd, const size_t *b_dims,
                        const ssize_t *b_str) {
  cuda_context *ctx = input->ctx;
  void *args[2];
  int res = GA_SYS_ERROR;
  int in_cache = 1;
  unsigned int i;
  size_t nEls = 1, ls[2], gs[2];
  gpukernel *k;
  cache_val_t *v;
  cache_key_t a;

  ASSERT_BUF(input);
  ASSERT_BUF(output);
  if (input->ctx != output->ctx)
    return GA_INVALID_ERROR;

  for (i = 0; i < a_nd; i++) {
    nEls *= a_dims[i];
  }
  if (nEls == 0) return GA_NO_ERROR;

  a.ind = a_nd;
  a.ond = b_nd;
  a.itype = intype;
  a.otype = outtype;
  a.ioff = ioff;
  a.ooff = ooff;
  a.idims = a_dims;
  a.odims = b_dims;
  a.istr = a_str;
  a.ostr = b_str;

  do_key_hash(&a);

  v = cache_get(ctx->extcopy_cache, &a);
  if (v == NULL) {
    v = &k;
    res = gen_extcopy_kernel(&a, input->ctx, v, nEls);
    if (res != GA_NO_ERROR)
      return res;

    /* Cache the kernel */
    a.idims = memdup(a_dims, a_nd*sizeof(size_t));
    a.odims = memdup(b_dims, b_nd*sizeof(size_t));
    a.istr = memdup(a_str, a_nd*sizeof(ssize_t));
    a.ostr = memdup(b_str, b_nd*sizeof(ssize_t));
    if (a.idims == NULL || a.odims == NULL ||
	a.istr == NULL || a.ostr == NULL ||
	cache_insert(ctx->extcopy_cache, &a, v)) {
      /* Cache insert or memdup failed */
      free((void *)a.idims);
      free((void *)a.odims);
      free((void *)a.istr);
      free((void *)a.ostr);
      in_cache = 0;
    }
  }

  /* Cheap kernel scheduling */
  res = cuda_property(NULL, NULL, *v, GA_KERNEL_PROP_MAXLSIZE, ls);
  if (res != GA_NO_ERROR) goto fail;

  gs[0] = ((nEls-1) / ls[0]) + 1;
  gs[1] = ls[1] = 1;
  args[0] = input;
  args[1] = output;
  res = cuda_callkernel(*v, ls, gs, args);

fail:
  if (!in_cache)
    cuda_freekernel(*v);
  return res;
}

static gpudata *cuda_transfer(gpudata *src, size_t offset, size_t sz,
                              void *dst_c, int may_share) {
  cuda_context *ctx = src->ctx;
  cuda_context *dst_ctx = (cuda_context *)dst_c;
  gpudata *dst;

  ASSERT_BUF(src);
  ASSERT_CTX(ctx);
  ASSERT_CTX(dst_ctx);

  if (ctx == dst_ctx) {
    if (may_share && offset == 0) {
        cuda_retain(src);
        return src;
    }
    dst = cuda_alloc(ctx, sz, NULL, src->flags & (GA_BUFFER_READ_ONLY|
                                                  GA_BUFFER_WRITE_ONLY), NULL);
    if (dst == NULL) return NULL;
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_free(dst);
      return NULL;
    }
    ctx->err = cuMemcpyDtoDAsync(dst->ptr, src->ptr+offset, sz, ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      cuda_free(dst);
      return NULL;
    }
    cuda_exit(ctx);
    return dst;
  }

  dst = cuda_alloc(dst_ctx, sz, NULL, src->flags & (GA_BUFFER_READ_ONLY|
                                                    GA_BUFFER_WRITE_ONLY),
                   NULL);
  if (dst == NULL)
    return NULL;
  cuda_enter(ctx);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_free(dst);
    return NULL;
  }
  ctx->err = cuMemcpyPeerAsync(dst->ptr, dst->ctx->ctx, src->ptr+offset,
			       src->ctx->ctx, sz, dst_ctx->s);
  cuEventRecord(dst->ev, dst_ctx->s);
  cuStreamWaitEvent(ctx->s, dst->ev, 0);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_free(dst);
    cuda_exit(ctx);
    return NULL;
  }
  cuda_exit(ctx);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_free(dst);
    return NULL;
  }
  return NULL;
}

#ifdef WITH_CUDA_CUBLAS
extern gpuarray_blas_ops cublas_ops;
#endif

static int cuda_property(void *c, gpudata *buf, gpukernel *k, int prop_id,
                         void *res) {
  cuda_context *ctx = NULL;
  if (c != NULL) {
    ctx = (cuda_context *)c;
    ASSERT_CTX(ctx);
  } else if (buf != NULL) {
    ASSERT_BUF(buf);
    ctx = buf->ctx;
  } else if (k != NULL) {
    ASSERT_KER(k);
    ctx = k->ctx;
  }
  /* I know that 512 and 1024 are magic numbers.
     There is an indication in buffer.h, though. */
  if (prop_id < 512) {
    if (ctx == NULL)
      return GA_VALUE_ERROR;
  } else if (prop_id < 1024) {
    if (buf == NULL)
      return GA_VALUE_ERROR;
  } else {
    if (k == NULL)
      return GA_VALUE_ERROR;
  }

  switch (prop_id) {
    char *s;
    CUdevice id;
    int i;

  case GA_CTX_PROP_DEVNAME:
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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

  case GA_CTX_PROP_BLAS_OPS:
#ifdef WITH_CUDA_CUBLAS
    *((gpuarray_blas_ops **)res) = &cublas_ops;
    return GA_NO_ERROR;
#else
    *((void **)res) = NULL;
    return GA_DEVSUP_ERROR;
#endif

  case GA_CTX_PROP_BIN_ID:
    *((const char **)res) = ctx->bin_id;
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_REFCNT:
    *((unsigned int *)res) = buf->refcnt;
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_SIZE:
    *((size_t *)res) = buf->sz;
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_CTX:
  case GA_KERNEL_PROP_CTX:
    *((void **)res) = (void *)ctx;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_MAXLSIZE:
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
    ctx->err = cuFuncGetAttribute(&i,
                                  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                  k->k);
    cuda_exit(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
    *((size_t *)res) = i;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_PREFLSIZE:
    cuda_enter(ctx);
    if (ctx->err != CUDA_SUCCESS)
      return GA_IMPL_ERROR;
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
    cuda_exit(ctx);
    *((size_t *)res) = i;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_NUMARGS:
    *((unsigned int *)res) = k->argcount;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_TYPES:
    *((const int **)res) = k->types;
    return GA_NO_ERROR;

  default:
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

GPUARRAY_LOCAL
const gpuarray_buffer_ops cuda_ops = {cuda_init,
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
                                     cuda_callkernel,
                                     cuda_kernelbin,
                                     cuda_sync,
                                     cuda_extcopy,
                                     cuda_transfer,
                                     cuda_property,
                                     cuda_error};
