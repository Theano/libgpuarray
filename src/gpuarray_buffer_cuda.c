#define _CRT_SECURE_NO_WARNINGS

#include "private.h"
#include "private_cuda.h"
#include "loaders/libnvrtc.h"
#include "loaders/libcublas.h"

#include <sys/types.h>

#include <assert.h>
#include <stdlib.h>

#include <cache.h>

#include "util/strb.h"
#include "util/xxhash.h"

#include "gpuarray/buffer.h"
#include "gpuarray/util.h"
#include "gpuarray/error.h"
#include "gpuarray/buffer_blas.h"

#include "gpuarray/extension.h"

STATIC_ASSERT(DONTFREE == GPUARRAY_CUDA_CTX_NOFREE, cuda_nofree_eq);
STATIC_ASSERT(CUDA_WAIT_READ == GPUARRAY_CUDA_WAIT_READ, cuda_wait_read_eq);
STATIC_ASSERT(CUDA_WAIT_WRITE == GPUARRAY_CUDA_WAIT_WRITE, cuda_wait_write_eq);
STATIC_ASSERT(sizeof(GpuArrayIpcMemHandle) == sizeof(CUipcMemHandle), cuda_ipcmem_eq);

/* Allocations will be made in blocks of at least this size */
#define BLOCK_SIZE (4 * 1024 * 1024)

/* No returned allocations will be smaller than this size.  Also, they
 * will be aligned to this size.
 *
 * Some libraries depend on this value and will crash if it's smaller.
 */
#define FRAG_SIZE (64)

const gpuarray_buffer_ops cuda_ops;

static void cuda_freekernel(gpukernel *);
static int cuda_property(gpucontext *, gpudata *, gpukernel *, int, void *);
static int cuda_waits(gpudata *, int, CUstream);
static int cuda_records(gpudata *, int, CUstream);

static int detect_arch(const char *prefix, char *ret, error *e);
static gpudata *new_gpudata(cuda_context *ctx, CUdeviceptr ptr, size_t size);

typedef struct _kernel_key {
  uint8_t version;
  uint8_t debug;
  uint8_t major;
  uint8_t minor;
  uint32_t reserved;
  char bin_id[64];
  strb src;
} kernel_key;

/* Size of the kernel_key that we can memcopy to duplicate */
#define KERNEL_KEY_MM (sizeof(kernel_key) - sizeof(strb))

static void key_free(cache_key_t _k) {
  kernel_key *k = (kernel_key *)_k;
  strb_clear(&k->src);
  free(k);
}

static int strb_eq(strb *k1, strb *k2) {
  return (k1->l == k2->l &&
          memcmp(k1->s, k2->s, k1->l) == 0);
}

static uint32_t strb_hash(strb *k) {
  return XXH32(k->s, k->l, 42);
}

static int key_eq(kernel_key *k1, kernel_key *k2) {
  return (memcmp(k1, k2, KERNEL_KEY_MM) == 0 &&
          strb_eq(&k1->src, &k2->src));
}

static int key_hash(kernel_key *k) {
  XXH32_state_t state;
  XXH32_reset(&state, 42);
  XXH32_update(&state, k, KERNEL_KEY_MM);
  XXH32_update(&state, k->src.s, k->src.l);
  return XXH32_digest(&state);
}

static int key_write(strb *res, kernel_key *k) {
  strb_appendn(res, (const char *)k, KERNEL_KEY_MM);
  strb_appendb(res, &k->src);
  return strb_error(res);
}

static kernel_key *key_read(const strb *b) {
  kernel_key *k;
  if (b->l < KERNEL_KEY_MM) return NULL;
  k = calloc(1, sizeof(*k));
  if (k == NULL) return NULL;
  memcpy(k, b->s, KERNEL_KEY_MM);
  if (k->version != 0) {
    free(k);
    return NULL;
  }
  if (strb_ensure(&k->src, b->l - KERNEL_KEY_MM) != 0) {
    strb_clear(&k->src);
    free(k);
    return NULL;
  }
  strb_appendn(&k->src, b->s + KERNEL_KEY_MM, b->l - KERNEL_KEY_MM);
  return k;
}

static int kernel_write(strb *res, strb *bin) {
  strb_appendb(res, bin);
  return strb_error(res);
}

static strb *kernel_read(const strb *b) {
  strb *res = strb_alloc(b->l);
  if (res != NULL)
    strb_appendb(res, b);
  return res;
}

static int setup_done = 0;
static int major = -1;
static int minor = -1;
static int setup_lib(void) {
  const char *ver;
  CUresult err;
  int res, tmp;

  if (!setup_done) {
    res = load_libcuda(global_err);
    if (res != GA_NO_ERROR)
      return res;
    err = cuInit(0);
    if (err != CUDA_SUCCESS)
      return error_cuda(global_err, "cuInit", err);
    ver = getenv("GPUARRAY_CUDA_VERSION");
    if (ver == NULL || strlen(ver) != 2) {
      err = cuDriverGetVersion(&tmp);
      if (err != CUDA_SUCCESS)
        return error_set(global_err, GA_IMPL_ERROR, "cuDriverGetVersion failed");
      major = tmp / 1000;
      minor = (tmp / 10) % 10;
    } else {
      major = ver[0] - '0';
      minor = ver[1] - '0';
    }
    if (major > 9 || major < 0 || minor > 9 || minor < 0)
      return error_fmt(global_err, GA_VALUE_ERROR, "Invalid cuda version: %d.%d", major, minor);
    res = load_libnvrtc(major, minor, global_err);
    if (res != GA_NO_ERROR)
      return res;
    setup_done = 1;
  }
  return GA_NO_ERROR;
}

static int cuda_get_platform_count(unsigned int* platcount) {
  *platcount = 1;  // CUDA works on NVIDIA's GPUs
  return GA_NO_ERROR;
}

static int cuda_get_device_count(unsigned int platform,
                                 unsigned int* devcount) {
  CUresult err;
  int dv;
  // platform number gets ignored in CUDA implementation
  GA_CHECK(setup_lib());
  err = cuDeviceGetCount(&dv);
  if (err != CUDA_SUCCESS)
    return error_cuda(global_err, "cuDeviceGetCount", err);
  *devcount = (unsigned int)dv;
  return GA_NO_ERROR;
}

cuda_context *cuda_make_ctx(CUcontext ctx, int flags) {
  cuda_context *res;
  cache *mem_cache;
  char *cache_path;
  void *p;
  CUresult err;
  int e;

  e = setup_lib();
  if (e != GA_NO_ERROR)
    return NULL;

  res = calloc(1, sizeof(*res));
  if (res == NULL) {
    error_sys(global_err, "calloc");
    return NULL;
  }
  res->ctx = ctx;
  res->ops = &cuda_ops;
  res->refcnt = 1;
  res->flags = flags;
  res->enter = 0;
  res->major = major;
  res->minor = minor;
  res->freeblocks = NULL;
  if (error_alloc(&res->err)) {
    error_set(global_err, GA_SYS_ERROR, "Could not create error context");
    goto fail_errmsg;
  }
  if (detect_arch(ARCH_PREFIX, res->bin_id, global_err)) {
    goto fail_stream;
  }
  /* Don't add the nonblocking flags to help usage with other
     libraries that may do stuff on the NULL stream */
  err = cuStreamCreate(&res->s, 0);
  if (err != CUDA_SUCCESS) {
    error_cuda(global_err, "cuStreamCreate", err);
    goto fail_stream;
  }
  if (ISSET(res->flags, GA_CTX_SINGLE_STREAM)) {
    res->mem_s = res->s;
  } else {
    /* Don't add the nonblocking flags to help usage with other
       libraries that may do stuff on the NULL stream */
    err = cuStreamCreate(&res->mem_s, 0);
    if (err != CUDA_SUCCESS) {
      error_cuda(global_err, "cuStreamCreate", err);
      goto fail_mem_stream;
    }
  }

  res->kernel_cache = cache_twoq(64, 128, 64, 8,
                                 (cache_eq_fn)strb_eq,
                                 (cache_hash_fn)strb_hash,
                                 (cache_freek_fn)strb_free,
                                 (cache_freev_fn)cuda_freekernel, global_err);
  if (res->kernel_cache == NULL) {
    error_cuda(global_err, "cuStreamCreate", err);
    goto fail_cache;
  }

  cache_path = getenv("GPUARRAY_CACHE_PATH");
  if (cache_path != NULL) {
    mem_cache = cache_lru(64, 8,
                          (cache_eq_fn)key_eq,
                          (cache_hash_fn)key_hash,
                          (cache_freek_fn)key_free,
                          (cache_freev_fn)strb_free,
                          global_err);
    if (mem_cache == NULL) {
      fprintf(stderr, "Error initializing mem cache for disk: %s\n",
              global_err->msg);
      goto fail_disk_cache;
    }
    res->disk_cache = cache_disk(cache_path, mem_cache,
                                 (kwrite_fn)key_write,
                                 (vwrite_fn)kernel_write,
                                 (kread_fn)key_read,
                                 (vread_fn)kernel_read,
                                 res->err);
    if (res->disk_cache == NULL) {
      // TODO use better error messages when they are available.
      fprintf(stderr, "Error initializing disk cache, disabling\n");
      cache_destroy(mem_cache);
      goto fail_disk_cache;
    }
  } else {
  fail_disk_cache:
    res->disk_cache = NULL;
  }

  err = cuMemAllocHost(&p, 16);
  if (err != CUDA_SUCCESS) {
    error_cuda(global_err, "cuMemAllocHost", err);
    goto fail_errbuf;
  }
  memset(p, 0, 16);
  /* Need to tag for new_gpudata */
  TAG_CTX(res);
  res->errbuf = new_gpudata(res, (CUdeviceptr)p, 16);
  if (res->errbuf == NULL) {
    /* Copy the error from the context since we are getting rid of it */
    error_set(global_err, res->err->code, res->err->msg);
    goto fail_end;
  }
  res->errbuf->flags |= CUDA_MAPPED_PTR;
  return res;
 fail_end:
  cuMemFreeHost(p);
 fail_errbuf:
  if (res->disk_cache)
    cache_destroy(res->disk_cache);
  cache_destroy(res->kernel_cache);
 fail_cache:
  if (ISCLR(res->flags, GA_CTX_SINGLE_STREAM))
    cuStreamDestroy(res->mem_s);
 fail_mem_stream:
  cuStreamDestroy(res->s);
 fail_stream:
  error_free(res->err);
 fail_errmsg:
  free(res);
  return NULL;
}

static void deallocate(gpudata *);

static void cuda_free_ctx(cuda_context *ctx) {
  gpuarray_blas_ops *blas_ops;
  gpudata *next, *curr;
  CUdevice dev;

  ASSERT_CTX(ctx);
  ctx->refcnt--;
  if (ctx->refcnt == 0) {
    assert(ctx->enter == 0 && "Context was active when freed!");
    if (ctx->blas_handle != NULL) {
      cuda_property((gpucontext *)ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS,
                    &blas_ops);
      blas_ops->teardown((gpucontext *)ctx);
    }
    cuMemFreeHost((void *)ctx->errbuf->ptr);
    deallocate(ctx->errbuf);

    if (ISCLR(ctx->flags, GA_CTX_SINGLE_STREAM))
      cuStreamDestroy(ctx->mem_s);
    cuStreamDestroy(ctx->s);

    /* Clear out the freelist */
    for (curr = ctx->freeblocks; curr != NULL; curr = next) {
      next = curr->next;
      cuMemFree(curr->ptr);
      deallocate(curr);
    }
    cache_destroy(ctx->kernel_cache);
    if (ctx->disk_cache)
      cache_destroy(ctx->disk_cache);
    error_free(ctx->err);

    if (!(ctx->flags & DONTFREE)) {
      cuCtxPushCurrent(ctx->ctx);
      cuCtxGetDevice(&dev);
      cuCtxPopCurrent(NULL);
      cuDevicePrimaryCtxRelease(dev);
    }
    CLEAR(ctx);
    free(ctx);
  }
}

CUstream cuda_get_stream(cuda_context *ctx) {
  ASSERT_CTX(ctx);
  return ctx->s;
}

void cuda_enter(cuda_context *ctx) {
  ASSERT_CTX(ctx);
  if (!ctx->enter)
    cuCtxPushCurrent(ctx->ctx);
  ctx->enter++;
}

void cuda_exit(cuda_context *ctx) {
  ASSERT_CTX(ctx);
  assert(ctx->enter > 0);
  ctx->enter--;
  if (!ctx->enter)
    cuCtxPopCurrent(NULL);
}

static gpudata *new_gpudata(cuda_context *ctx, CUdeviceptr ptr, size_t size) {
  gpudata *res;
  CUresult err;
  int fl = CU_EVENT_DISABLE_TIMING;

  res = malloc(sizeof(*res));
  if (res == NULL) {
    error_sys(ctx->err, "malloc");
    return NULL;
  }

  res->refcnt = 0;
  res->sz = size;

  res->flags = 0;
  res->ls = NULL;

  cuda_enter(ctx);

  if (ctx->flags & GA_CTX_MULTI_THREAD)
    fl |= CU_EVENT_BLOCKING_SYNC;
  err = cuEventCreate(&res->rev, fl);
  if (err != CUDA_SUCCESS) {
    error_cuda(ctx->err, "cuEventCreate", err);
    cuda_exit(ctx);
    free(res);
    return NULL;
  }

  err = cuEventCreate(&res->wev, fl);
  if (err != CUDA_SUCCESS) {
    error_cuda(ctx->err, "cuEventCreate", err);
    cuEventDestroy(res->rev);
    cuda_exit(ctx);
    free(res);
    return NULL;
  }

  cuda_exit(ctx);

  res->ptr = ptr;
  res->next = NULL;
  res->ctx = ctx;
  TAG_BUF(res);

  return res;
}

gpudata *cuda_make_buf(cuda_context *ctx, CUdeviceptr p, size_t sz) {
  gpudata *res = new_gpudata(ctx, p, sz);

  if (res == NULL) return NULL;

  res->refcnt = 1;
  res->flags |= DONTFREE;
  res->ctx->refcnt++;

  return res;
}

size_t cuda_get_sz(gpudata *g) { ASSERT_BUF(g); return g->sz; }

#define CHKFAIL(e, n, v)      \
  if (err != CUDA_SUCCESS) { \
    error_cuda(e, n, err);   \
    return v;                \
  }

static const char CUDA_PREAMBLE[] =
    "#define local_barrier() __syncthreads()\n"
    "#define WITHIN_KERNEL extern \"C\" __device__\n"
    "#define KERNEL extern \"C\" __global__\n"
    "#define GLOBAL_MEM /* empty */\n"
    "#define LOCAL_MEM __shared__\n"
    "#define LOCAL_MEM_ARG /* empty */\n"
    "#define REQD_WG_SIZE(X,Y,Z) __launch_bounds__(X*Y, Z)\n"
    "#ifdef NAN\n"
    "#undef NAN\n"
    "#endif\n"
    "#define NAN __int_as_float(0x7fffffff)\n"
    "#ifdef INFINITY\n"
    "#undef INFINITY\n"
    "#endif\n"
    "#define INFINITY __int_as_float(0x7f800000)\n"
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
    "#define ga_bool unsigned char\n"
    "#define ga_byte signed char\n"
    "#define ga_ubyte unsigned char\n"
    "#define ga_short short\n"
    "#define ga_ushort unsigned short\n"
    "#define ga_int int\n"
    "#define ga_uint unsigned int\n"
    "#define ga_long long long\n"
    "#define ga_ulong unsigned long long\n"
    "#define ga_float float\n"
    "#define ga_double double\n"
    "#define ga_half ga_ushort\n"
    "#define ga_size size_t\n"
    "#define ga_ssize ptrdiff_t\n"
    "#define load_half(p) __half2float(*(p))\n"
    "#define store_half(p, v) (*(p) = __float2half_rn(v))\n"
    "#define GA_DECL_SHARED_PARAM(type, name)\n"
    "#define GA_DECL_SHARED_BODY(type, name) extern __shared__ type name[];\n"
    "#define GA_WARP_SIZE warpSize\n"
    "#line 1\n";

/* XXX: add complex, quads, longlong */
/* XXX: add vector types */

static cuda_context *do_init(CUdevice dev, int flags, error *e) {
  cuda_context *res;
  CUcontext ctx;
  CUresult err;
  unsigned int fl = CU_CTX_SCHED_AUTO;
  unsigned int cur_fl;
  int act;
  int i;

  if (flags & GA_CTX_SINGLE_THREAD)
    fl = CU_CTX_SCHED_SPIN;
  if (flags & GA_CTX_MULTI_THREAD)
    fl = CU_CTX_SCHED_YIELD;
  err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
  CHKFAIL(e, "cuDeviceGetAttribute", NULL);
  if (i != 1) {
    error_set(e, GA_UNSUPPORTED_ERROR, "device does not support unified addressing");
    return NULL;
  }
  err = cuDevicePrimaryCtxGetState(dev, &cur_fl, &act);
  CHKFAIL(e, "cuDevicePrimaryCtxGetState", NULL);
  if (act == 1) {
    if ((cur_fl & fl) != fl) {
      error_set(e, GA_INVALID_ERROR, "device is already active and has unsupported flags");
      return NULL;
    }
  } else {
    err = cuDevicePrimaryCtxSetFlags(dev, fl);
    CHKFAIL(e, "cuDevicePrimaryCtxSetFlags", NULL);
  }
  err = cuDevicePrimaryCtxRetain(&ctx, dev);
  CHKFAIL(e, "cuDevicePrimaryCtxRetain", NULL);
  err = cuCtxPushCurrent(ctx);
  CHKFAIL(e, "cuCtxPushCurrent", NULL);
  res = cuda_make_ctx(ctx, flags);
  if (res == NULL) {
    cuDevicePrimaryCtxRelease(dev);
    if (e != global_err)
      error_set(e, global_err->code, global_err->msg);
    return NULL;
  }
  /* Don't leave the context on the thread stack */
  cuCtxPopCurrent(NULL);

  return res;
}

static gpucontext *cuda_init(int ord, int flags) {
    CUdevice dev;
    cuda_context *res;
    CUresult err;
    int r;

    r = setup_lib();
    if (r != GA_NO_ERROR) {
      return NULL;
    }

    if (ord == -1) {
      int i, c;
      err = cuDeviceGetCount(&c);
      CHKFAIL(global_err, "cuDeviceGetCount", NULL);
      for (i = 0; i < c; i++) {
        err = cuDeviceGet(&dev, i);
        CHKFAIL(global_err, "cuDeviceGet", NULL);
        res = do_init(dev, flags, global_err);
        if (res != NULL)
          return (gpucontext *)res;
      }
      error_set(global_err, GA_NODEV_ERROR, "No cuda device available");
      return NULL;
    } else {
      err = cuDeviceGet(&dev, ord);
      CHKFAIL(global_err, "cuDeviceGet", NULL);
      return (gpucontext *)do_init(dev, flags, global_err);
    }
}
static void cuda_deinit(gpucontext *c) {
  cuda_free_ctx((cuda_context *)c);
}

/*
 * Find the block in the free list that is the best fit for the size
 * we want, which means the smallest that can still fit the size.
 */
static void find_best(cuda_context *ctx, gpudata **best, gpudata **prev,
                     size_t size) {
  gpudata *temp, *tempPrev = NULL;
  *best = NULL;

  for (temp = ctx->freeblocks; temp; temp = temp->next) {
    if (temp->sz >= size && (!*best || temp->sz < (*best)->sz)) {
      *best = temp;
      *prev = tempPrev;
    }
    tempPrev = temp;
  }
}

static size_t largest_size(cuda_context *ctx) {
  gpudata *temp;
  size_t sz, dummy;
  cuda_enter(ctx);
  cuMemGetInfo(&sz, &dummy);
  cuda_exit(ctx);
   /* We guess that we can allocate at least a quarter of the free size
     in a single block. This might be wrong though. */
  sz /= 4;
  for (temp = ctx->freeblocks; temp; temp = temp->next) {
    if (temp->sz > sz) sz = temp->sz;
  }
  return sz;
}

/*
 * Allocate a new block and place in on the freelist. Will allocate
 * the bigger of the requested size and BLOCK_SIZE to avoid allocating
 * multiple small blocks.
 */
static int allocate(cuda_context *ctx, gpudata **res, gpudata **prev,
                    size_t size) {
  CUdeviceptr ptr;
  gpudata *next;
  CUresult err;

  *prev = NULL;

  if (!(ctx->flags & GA_CTX_DISABLE_ALLOCATION_CACHE))
    if (size < BLOCK_SIZE) size = BLOCK_SIZE;

  cuda_enter(ctx);

  err = cuMemAlloc(&ptr, size);
  if (err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    return error_cuda(ctx->err, "cuMemAlloc", err);
  }

  *res = new_gpudata(ctx, ptr, size);

  cuda_exit(ctx);

  if (*res == NULL) {
    cuMemFree(ptr);
    return ctx->err->code;
  }

  (*res)->flags |= CUDA_HEAD_ALLOC;

  /* Now that the block is allocated, enter it in the freelist */
  next = ctx->freeblocks;
  for (; next && next->ptr < (*res)->ptr; next = next->next) {
    *prev = next;
  }
  (*res)->next = next;
  if (*prev)
    (*prev)->next = *res;
  else
    ctx->freeblocks = *res;

  return GA_NO_ERROR;
}

/*
 * Extract the `curr` block from the freelist, possibly splitting it
 * if it's too big for the requested size.  The remaining block will
 * stay on the freelist if there is a split.  `prev` is only to
 * facilitate the extraction so we don't have to go through the list
 * again.
 */
static int extract(gpudata *curr, gpudata *prev, size_t size) {
  gpudata *next, *split;
  size_t remaining = curr->sz - size;

  if (remaining < FRAG_SIZE) {
    /* No need to split, the remaining block would be too small */
    next = curr->next;
  } else {
    split = new_gpudata(curr->ctx, curr->ptr + size, remaining);
    if (split == NULL)
      return curr->ctx->err->code;
    /* Make sure the chain keeps going */
    split->next = curr->next;
    curr->next = NULL;
    /* Make sure we don't start using the split buffer too soon */
    cuda_records(split, CUDA_WAIT_ALL, curr->ls);
    next = split;
    curr->sz = size;
  }

  if (prev != NULL)
    prev->next = next;
  else
    curr->ctx->freeblocks = next;

  return GA_NO_ERROR;
}

static void cuda_free(gpudata *);
static int cuda_write(gpudata *dst, size_t dstoff, const void *src,
                      size_t sz);

static inline size_t roundup(size_t s, size_t m) {
  return ((s + (m - 1)) / m) * m;
}

static gpudata *cuda_alloc(gpucontext *c, size_t size, void *data, int flags) {
  gpudata *res = NULL, *prev = NULL;
  cuda_context *ctx = (cuda_context *)c;
  size_t asize;

  if ((flags & GA_BUFFER_INIT) && data == NULL) {
    error_set(ctx->err, GA_VALUE_ERROR, "Requested buffer initialisation but no data given");
    return NULL;
  }
  if ((flags & (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) ==
      (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) {
    error_set(ctx->err, GA_VALUE_ERROR, "Invalid flags combinaison WRITE_ONLY and READ_ONLY");
    return NULL;
  }

  /* TODO: figure out how to make this work */
  if (flags & GA_BUFFER_HOST) {
    error_set(ctx->err, GA_DEVSUP_ERROR, "Host mapped allocations are not supported yet");
    return NULL;
  }

  /* We don't want to manage really small allocations so we round up
   * to a multiple of FRAG_SIZE.  This also ensures that if we split a
   * block, the next block starts properly aligned for any data type.
   */
  if (!(ctx->flags & GA_CTX_DISABLE_ALLOCATION_CACHE)) {
    asize = roundup(size, FRAG_SIZE);
    find_best(ctx, &res, &prev, asize);
  } else {
    asize = size;
  }

  if (res == NULL && allocate(ctx, &res, &prev, asize) != GA_NO_ERROR)
    return NULL;

  if (extract(res, prev, asize) != GA_NO_ERROR)
    return NULL;

  /* It's out of the freelist, so add a ref */
  res->ctx->refcnt++;
  /* We consider this buffer allocated and ready to go */
  res->refcnt = 1;

  if (flags & GA_BUFFER_INIT) {
    if (cuda_write(res, 0, data, size) != GA_NO_ERROR) {
      cuda_free(res);
      return NULL;
    }
  }

  return res;
}

int cuda_get_ipc_handle(gpudata *d, GpuArrayIpcMemHandle *h) {
  ASSERT_BUF(d);
  cuda_enter(d->ctx);
  CUDA_EXIT_ON_ERROR(d->ctx,
                     cuIpcGetMemHandle((CUipcMemHandle *)h, d->ptr));
  cuda_exit(d->ctx);
  return GA_NO_ERROR;
}

gpudata *cuda_open_ipc_handle(gpucontext *c, GpuArrayIpcMemHandle *h, size_t sz) {
  CUdeviceptr p;
  cuda_context *ctx = (cuda_context *)c;
  gpudata *d = NULL;
  CUresult err;

  cuda_enter(ctx);
  err = cuIpcOpenMemHandle(&p, *((CUipcMemHandle *)h),
                           CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
  if (err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    error_cuda(ctx->err, "cuIpcOpenMemHandle", err);
    return NULL;
  }
  d = cuda_make_buf(ctx, p, sz);
  if (d != NULL)
    d->flags |= CUDA_IPC_MEMORY;
  return d;
}

static void cuda_retain(gpudata *d) {
  ASSERT_BUF(d);
  d->refcnt++;
}

static void deallocate(gpudata *d) {
  cuda_enter(d->ctx);
  cuEventDestroy(d->rev);
  cuEventDestroy(d->wev);
  cuda_exit(d->ctx);
  CLEAR(d);
  free(d);
}

static void cuda_free(gpudata *d) {
  /* We ignore errors on free */
  ASSERT_BUF(d);
  d->refcnt--;
  if (d->refcnt == 0) {
    /* Keep a reference to the context since we deallocate the gpudata
     * object */
    cuda_context *ctx = d->ctx;
    if (d->flags & DONTFREE) {
      /* This is the path for "external" buffers */
      deallocate(d);
    } else if (d->flags & CUDA_IPC_MEMORY) {
      cuIpcCloseMemHandle(d->ptr);
      deallocate(d);
    } else if (ctx->flags & GA_CTX_DISABLE_ALLOCATION_CACHE) {
      /* Just free the pointer */
      cuMemFree(d->ptr);
      deallocate(d);
    } else {
      /* Find the position in the freelist.  Freelist is kept in order
         of allocation address */
      gpudata *next = d->ctx->freeblocks, *prev = NULL;
      for (; next && next->ptr < d->ptr; next = next->next) {
        prev = next;
      }
      next = prev != NULL ? prev->next : d->ctx->freeblocks;

      /* See if we can merge the block with the previous one */
      if (!(d->flags & CUDA_HEAD_ALLOC) &&
            prev != NULL && prev->ptr + prev->sz == d->ptr) {
        prev->sz = prev->sz + d->sz;
        cuda_waits(d, CUDA_WAIT_ALL, prev->ls);
        cuda_records(prev, CUDA_WAIT_ALL, prev->ls);
        deallocate(d);
        d = prev;
      } else if (prev != NULL) {
        prev->next = d;
      } else {
        d->ctx->freeblocks = d;
      }

      /* See if we can merge with next */
      if (next && !(next->flags & CUDA_HEAD_ALLOC) &&
          d->ptr + d->sz == next->ptr) {
        d->sz = d->sz + next->sz;
        d->next = next->next;
        cuda_wait(next, CUDA_WAIT_ALL);
        cuda_record(d, CUDA_WAIT_ALL);
        deallocate(next);
      } else {
        d->next = next;
      }
    }
    /* We keep this at the end since the freed buffer could be the
     * last reference to the context and therefore clearing the
     * reference could trigger the freeing if the whole context
     * including the freelist, which we manipulate. */
    cuda_free_ctx(ctx);
  }
}

static int cuda_share(gpudata *a, gpudata *b) {
  ASSERT_BUF(a);
  ASSERT_BUF(b);
  return (a->ctx == b->ctx && a->sz != 0 && b->sz != 0 &&
          ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
           (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr)));
}

static int cuda_waits(gpudata *a, int flags, CUstream s) {
  ASSERT_BUF(a);

  /* Never skip the wait if CUDA_WAIT_FORCE */
  if (ISCLR(flags, CUDA_WAIT_FORCE)) {
    if (ISSET(a->ctx->flags, GA_CTX_SINGLE_STREAM))
      return GA_NO_ERROR;

    /* If the last stream to touch this buffer is the same, we don't
     * need to wait for anything. */
    if (a->ls == s)
      return GA_NO_ERROR;
  }

  cuda_enter(a->ctx);
  /* We wait for writes that happened before since multiple reads at
   * the same time are fine */
  if (ISSET(flags, CUDA_WAIT_READ) || ISSET(flags, CUDA_WAIT_WRITE))
    CUDA_EXIT_ON_ERROR(a->ctx, cuStreamWaitEvent(s, a->wev, 0));
  /* Make sure to not disturb previous reads */
  if (ISSET(flags, CUDA_WAIT_WRITE))
    CUDA_EXIT_ON_ERROR(a->ctx, cuStreamWaitEvent(s, a->rev, 0));
  cuda_exit(a->ctx);
  return GA_NO_ERROR;
}

int cuda_wait(gpudata *a, int flags) {
  return cuda_waits(a, flags, a->ctx->s);
}

static int cuda_records(gpudata *a, int flags, CUstream s) {
  ASSERT_BUF(a);
  if (ISCLR(flags, CUDA_WAIT_FORCE) &&
      ISSET(a->ctx->flags, GA_CTX_SINGLE_STREAM))
    return GA_NO_ERROR;
  cuda_enter(a->ctx);
  if (ISSET(flags, CUDA_WAIT_READ))
    CUDA_EXIT_ON_ERROR(a->ctx, cuEventRecord(a->rev, s));
  if (ISSET(flags, CUDA_WAIT_WRITE))
    CUDA_EXIT_ON_ERROR(a->ctx, cuEventRecord(a->wev, s));
  cuda_exit(a->ctx);
  a->ls = s;
  return GA_NO_ERROR;
}

int cuda_record(gpudata *a, int flags) {
  return cuda_records(a, flags, a->ctx->s);
}

static int cuda_move(gpudata *dst, size_t dstoff, gpudata *src,
                     size_t srcoff, size_t sz) {
    cuda_context *ctx = dst->ctx;
    int res = GA_NO_ERROR;
    ASSERT_BUF(dst);
    ASSERT_BUF(src);
    if (src->ctx != dst->ctx) return error_set(ctx->err, GA_VALUE_ERROR,
                                               "Cannot move between contexts");

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz)
      return error_set(ctx->err, GA_VALUE_ERROR, "Destination is smaller than requested transfer size");
    if ((src->sz - srcoff) < sz)
      return error_set(ctx->err, GA_VALUE_ERROR, "Source is smaller than requested transfer size");

    cuda_enter(ctx);

    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_wait(src, CUDA_WAIT_READ));
    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_wait(dst, CUDA_WAIT_WRITE));

    CUDA_EXIT_ON_ERROR(ctx,
        cuMemcpyDtoDAsync(dst->ptr + dstoff, src->ptr + srcoff, sz, ctx->s));

    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_record(src, CUDA_WAIT_READ));
    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_record(dst, CUDA_WAIT_WRITE));

    cuda_exit(ctx);
    return res;
}

static int cuda_read(void *dst, gpudata *src, size_t srcoff, size_t sz) {
    cuda_context *ctx = src->ctx;

    ASSERT_BUF(src);

    if (sz == 0) return GA_NO_ERROR;

    if ((src->sz - srcoff) < sz)
      return error_set(ctx->err, GA_VALUE_ERROR, "source is smaller than the read size");

    cuda_enter(ctx);

    if (src->flags & CUDA_MAPPED_PTR) {

      if (ISSET(ctx->flags, GA_CTX_SINGLE_STREAM))
        CUDA_EXIT_ON_ERROR(ctx, cuStreamSynchronize(ctx->s));
      else
        CUDA_EXIT_ON_ERROR(ctx, cuEventSynchronize(src->wev));

      memcpy(dst, (void *)(src->ptr + srcoff), sz);
    } else {
      GA_CUDA_EXIT_ON_ERROR(ctx,
          cuda_waits(src, CUDA_WAIT_READ, ctx->mem_s));

      CUDA_EXIT_ON_ERROR(ctx,
          cuMemcpyDtoHAsync(dst, src->ptr + srcoff, sz, ctx->mem_s));

      GA_CUDA_EXIT_ON_ERROR(ctx,
          cuda_records(src, CUDA_WAIT_READ, ctx->mem_s));
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
      return error_set(ctx->err, GA_VALUE_ERROR, "Destination is smaller than the write size");

    cuda_enter(ctx);

    if (dst->flags & CUDA_MAPPED_PTR) {

      if (ISSET(ctx->flags, GA_CTX_SINGLE_STREAM))
        CUDA_EXIT_ON_ERROR(ctx, cuStreamSynchronize(ctx->s));
      else
        CUDA_EXIT_ON_ERROR(ctx, cuEventSynchronize(dst->rev));

      memcpy((void *)(dst->ptr + dstoff), src, sz);
    } else {
      GA_CUDA_EXIT_ON_ERROR(ctx,
          cuda_waits(dst, CUDA_WAIT_WRITE, ctx->mem_s));

      CUDA_EXIT_ON_ERROR(ctx,
          cuMemcpyHtoDAsync(dst->ptr + dstoff, src, sz, ctx->mem_s));

      GA_CUDA_EXIT_ON_ERROR(ctx,
          cuda_records(dst, CUDA_WAIT_WRITE, ctx->mem_s));
    }
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, size_t dstoff, int data) {
    cuda_context *ctx = dst->ctx;

    ASSERT_BUF(dst);

    if ((dst->sz - dstoff) == 0) return GA_NO_ERROR;

    cuda_enter(ctx);

    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_wait(dst, CUDA_WAIT_WRITE));

    CUDA_EXIT_ON_ERROR(ctx,
        cuMemsetD8Async(dst->ptr + dstoff, data, dst->sz - dstoff, ctx->s));

    GA_CUDA_EXIT_ON_ERROR(ctx,
        cuda_record(dst, CUDA_WAIT_WRITE));
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int get_cc(CUdevice dev, int *maj, int *min, error *e) {
  CUresult err;
  err = cuDeviceGetAttribute(maj,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                             dev);
  if (err != CUDA_SUCCESS)
    return error_cuda(e, "cuDeviceGetAttribute", err);
  err = cuDeviceGetAttribute(min,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                             dev);
  if (err != CUDA_SUCCESS)
    return error_cuda(e, "cuDeviceGetAttribute", err);
  return GA_NO_ERROR;
}

static int detect_arch(const char *prefix, char *ret, error *e) {
  CUdevice dev;
  CUresult err;
  int major, minor;
  int res;
  size_t sz = strlen(prefix) + 3;
  err = cuCtxGetDevice(&dev);
  if (err != CUDA_SUCCESS) return error_cuda(e, "cuCtxGetDevice", err);
  GA_CHECK(get_cc(dev, &major, &minor, e));
  res = snprintf(ret, sz, "%s%d%d", prefix, major, minor);
  if (res == -1) return error_sys(e, "snprintf");
  if (res > (ssize_t)sz) return error_set(e, GA_UNSUPPORTED_ERROR,
                                          "detect_arch: arch id is too large");
  return GA_NO_ERROR;
}

static inline int error_nvrtc(error *e, const char *msg, nvrtcResult err) {
  return error_fmt(e, GA_IMPL_ERROR, "%s: %s", msg, nvrtcGetErrorString(err));
}

static int call_compiler(cuda_context *ctx, strb *src, strb *ptx, strb *log) {
  nvrtcProgram prog;
  size_t buflen;
  const char *opts[4] = {
    "-arch", ""
    , "-G", "-lineinfo"
  };
  nvrtcResult err;

  opts[1] = ctx->bin_id;

  err = nvrtcCreateProgram(&prog, src->s, NULL, 0, NULL, NULL);
  if (err != NVRTC_SUCCESS)
    return error_nvrtc(ctx->err, "nvrtcCreateProgram", err);

  err = nvrtcCompileProgram(prog,
#ifdef DEBUG
                            4,
#else
                            2,
#endif
                            opts);

  /* Get the log before handling the error */
  if (nvrtcGetProgramLogSize(prog, &buflen) == NVRTC_SUCCESS) {
    strb_appends(log, "NVRTC compile log::\n");
    if (strb_ensure(log, buflen) == 0)
      if (nvrtcGetProgramLog(prog, log->s+log->l) == NVRTC_SUCCESS)
        log->l += buflen - 1; // Remove the final NUL
    strb_appendc(log, '\n');
  }

  if (err != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return error_nvrtc(ctx->err, "nvrtcCompileProgram", err);
  }

  err = nvrtcGetPTXSize(prog, &buflen);
  if (err != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    return error_nvrtc(ctx->err, "nvrtcGetPTXSize", err);
  }

  if (strb_ensure(ptx, buflen) == 0) {
    err = nvrtcGetPTX(prog, ptx->s+ptx->l);
    if (err != NVRTC_SUCCESS) {
      nvrtcDestroyProgram(&prog);
      return error_nvrtc(ctx->err, "nvrtcGetPTX", err);
    }
    ptx->l += buflen;
  }

  return GA_NO_ERROR;
}

static int make_bin(cuda_context *ctx, const strb *ptx, strb *bin, strb *log) {
  char info_log[2048] = "";
  char error_log[2048] = "";
  void *out;
  size_t out_size;
  CUlinkState st;
  CUjit_option cujit_opts[] = {
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_LOG_VERBOSE,
    CU_JIT_GENERATE_DEBUG_INFO,
    CU_JIT_GENERATE_LINE_INFO,
  };
  void *cujit_opt_vals[] = {
    (void *)sizeof(info_log), info_log,
    (void *)sizeof(error_log), error_log,
#ifdef DEBUG
    (void *)1, (void *)1, (void *)1
#else
    (void *)0, (void *)0, (void *)0
#endif
  };
  CUresult err;
  int res = GA_NO_ERROR;

  err = cuLinkCreate(sizeof(cujit_opts)/sizeof(cujit_opts[0]),
                          cujit_opts, cujit_opt_vals, &st);
  if (err != CUDA_SUCCESS)
    return error_cuda(ctx->err, "cuLinkCreate", err);
  err = cuLinkAddData(st, CU_JIT_INPUT_PTX, ptx->s, ptx->l,
                           "kernel code", 0, NULL, NULL);
  if (err != CUDA_SUCCESS) {
    res = error_cuda(ctx->err, "cuLinkAddData", err);
    goto out;
  }
  err = cuLinkComplete(st, &out, &out_size);
  if (err != CUDA_SUCCESS) {
    res = error_cuda(ctx->err, "cuLinkComplete", err);
    goto out;
  }
  strb_appendn(bin, out, out_size);
out:
  cuLinkDestroy(st);
  strb_appends(log, "Link info log::\n");
  strb_appends(log, info_log);
  strb_appends(log, "\nLink error log::\n");
  strb_appends(log, error_log);
  strb_appendc(log, '\n');
  return res;
}

static int compile(cuda_context *ctx, strb *src, strb* bin, strb *log) {
  strb ptx = STRB_STATIC_INIT;
  strb *cbin;
  kernel_key k;
  kernel_key *pk;

  memset(&k, 0, sizeof(k));
  k.version = 0;
#ifdef DEBUG
  k.debug = 1;
#endif
  k.major = ctx->major;
  k.minor = ctx->minor;
  memcpy(k.bin_id, ctx->bin_id, 64);
  memcpy(&k.src, src, sizeof(strb));

  // Look up the binary in the disk cache
  if (ctx->disk_cache) {
    cbin = cache_get(ctx->disk_cache, &k);
    if (cbin != NULL) {
      strb_appendb(bin, cbin);
      return GA_NO_ERROR;
    }
  }

  GA_CHECK(call_compiler(ctx, src, &ptx, log));

  GA_CHECK(make_bin(ctx, &ptx, bin, log));

  if (ctx->disk_cache) {
    pk = calloc(sizeof(kernel_key), 1);
    if (pk == NULL) {
      error_sys(ctx->err, "calloc");
      fprintf(stderr, "Error adding kernel to disk cache: %s\n",
              ctx->err->msg);
      return GA_NO_ERROR;
    }
    memcpy(pk, &k, KERNEL_KEY_MM);
    strb_appendb(&pk->src, src);
    if (strb_error(&pk->src)) {
      error_sys(ctx->err, "strb_appendb"); 
      fprintf(stderr, "Error adding kernel to disk cache %s\n",
              ctx->err->msg);
      key_free((cache_key_t)pk);
      return GA_NO_ERROR;
    }
    cbin = strb_alloc(bin->l);
    if (cbin == NULL) {
      error_sys(ctx->err, "strb_alloc"); 
      fprintf(stderr, "Error adding kernel to disk cache: %s\n",
              ctx->err->msg);
      key_free((cache_key_t)pk);
      return GA_NO_ERROR;
    }
    strb_appendb(cbin, bin);
    if (strb_error(cbin)) {
      error_sys(ctx->err, "strb_appendb"); 
      fprintf(stderr, "Error adding kernel to disk cache %s\n",
              ctx->err->msg);
      key_free((cache_key_t)pk);
      strb_free(cbin);
      return GA_NO_ERROR;
    }
    if (cache_add(ctx->disk_cache, pk, cbin)) {
      // TODO use better error messages
      fprintf(stderr, "Error adding kernel to disk cache\n");
    }
  }

  return GA_NO_ERROR;
}

static void _cuda_freekernel(gpukernel *k) {
  k->refcnt--;
  if (k->refcnt == 0) {
    if (k->ctx != NULL) {
      cuda_enter(k->ctx);
      cuModuleUnload(k->m);
      cuda_exit(k->ctx);
      cuda_free_ctx(k->ctx);
    }
    CLEAR(k);
    free(k->args);
    free(k->bin);
    free(k->types);
    free(k);
  }
}

static gpukernel *cuda_newkernel(gpucontext *c, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, unsigned int argcount,
                                 const int *types, int flags, char **err_str) {
    cuda_context *ctx = (cuda_context *)c;
    strb src = STRB_STATIC_INIT;
    strb bin = STRB_STATIC_INIT;
    strb log = STRB_STATIC_INIT;
    strb *psrc;
    gpukernel *res;
    CUdevice dev;
    CUresult err;
    unsigned int i;
    int major, minor;

    if (count == 0) {
      error_set(ctx->err, GA_VALUE_ERROR, "String count is 0");
      return NULL;
    }

    if (flags & GA_USE_OPENCL) {
      error_set(ctx->err, GA_DEVSUP_ERROR, "OpenCL kernel not supported on cuda devices");
      return NULL;
    }

    if (flags & GA_USE_BINARY) {
      error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Binary mode not supported any more");
      return NULL;
    }

    cuda_enter(ctx);

    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      error_cuda(ctx->err, "cuCtxGetDevice", err);
      return NULL;
    }

    if (get_cc(dev, &major, &minor, ctx->err) != GA_NO_ERROR)
      return NULL;

    // GA_USE_CLUDA is done later
    // GA_USE_SMALL will always work
    // GA_USE_HALF should always work
    if (flags & GA_USE_DOUBLE) {
      if (major < 1 || (major == 1 && minor < 3)) {
        cuda_exit(ctx);
        error_set(ctx->err, GA_DEVSUP_ERROR, "Requested double support and current device doesn't support them");
        return NULL;
      }
    }
    if (flags & GA_USE_COMPLEX) {
      // just for now since it is most likely broken
      cuda_exit(ctx);
      error_set(ctx->err, GA_UNSUPPORTED_ERROR, "Complex support is not there yet.");
    }

    if (flags & GA_USE_CLUDA) {
      strb_appends(&src, CUDA_PREAMBLE);
    }

    if (lengths == NULL) {
      for (i = 0; i < count; i++)
        strb_appends(&src, strings[i]);
    } else {
      for (i = 0; i < count; i++) {
        if (lengths[i] == 0)
          strb_appends(&src, strings[i]);
        else
          strb_appendn(&src, strings[i], lengths[i]);
      }
    }

    strb_append0(&src);

    if (strb_error(&src)) {
      error_sys(ctx->err, "strb");
      strb_clear(&src);
      cuda_exit(ctx);
      return NULL;
    }

    res = (gpukernel *)cache_get(ctx->kernel_cache, &src);
    if (res != NULL) {
      res->refcnt++;
      strb_clear(&src);
      return res;
    }

    if (compile(ctx, &src, &bin, &log) != GA_NO_ERROR) {
      if (err_str != NULL) {
        strb debug_msg = STRB_STATIC_INIT;
        strb_appends(&debug_msg, "CUDA kernel compile failure ::\n");
        src.l--;
        gpukernel_source_with_line_numbers(1, (const char **)&src.s,
                                           &src.l, &debug_msg);
        strb_appends(&debug_msg, "\nCompile log:\n");
        strb_appendb(&debug_msg, &log);
        *err_str = strb_cstr(&debug_msg);
      }
      strb_clear(&src);
      strb_clear(&bin);
      strb_clear(&log);
      cuda_exit(ctx);
      return NULL;
    }
    strb_clear(&log);

    if (strb_error(&bin)) {
      error_sys(ctx->err, "strb");
      strb_clear(&src);
      strb_clear(&bin);
      cuda_exit(ctx);
      return NULL;
    }

    res = calloc(1, sizeof(*res));
    if (res == NULL) {
      error_sys(ctx->err, "calloc");
      strb_clear(&src);
      strb_clear(&bin);
      cuda_exit(ctx);
      return NULL;
    }

    /* Don't clear bin after this */
    res->bin_sz = bin.l;
    res->bin = bin.s;
    res->refcnt = 1;
    res->argcount = argcount;
    res->types = calloc(argcount, sizeof(int));
    if (res->types == NULL) {
      error_sys(ctx->err, "calloc");
      _cuda_freekernel(res);
      strb_clear(&src);
      cuda_exit(ctx);
      return NULL;
    }
    memcpy(res->types, types, argcount*sizeof(int));
    res->args = calloc(argcount, sizeof(void *));
    if (res->args == NULL) {
      error_sys(ctx->err, "calloc");
      _cuda_freekernel(res);
      strb_clear(&src);
      cuda_exit(ctx);
      return NULL;
    }

    err = cuModuleLoadData(&res->m, bin.s);
    if (err != CUDA_SUCCESS) {
      error_cuda(ctx->err, "cuModuleLoadData", err);
      _cuda_freekernel(res);
      strb_clear(&src);
      cuda_exit(ctx);
      return NULL;
    }

    err = cuModuleGetFunction(&res->k, res->m, fname);
    if (err != CUDA_SUCCESS) {
      error_cuda(ctx->err, "cuModuleGetFunction", err);
      _cuda_freekernel(res);
      strb_clear(&src);
      cuda_exit(ctx);
      return NULL;
    }

    res->ctx = ctx;
    ctx->refcnt++;
    cuda_exit(ctx);
    TAG_KER(res);
    psrc = memdup(&src, sizeof(strb));
    if (psrc != NULL) {
      /* One of the refs is for the cache */
      res->refcnt++;
      /* If this fails, it will free the key and remove a ref from the
         kernel. */
      cache_add(ctx->kernel_cache, psrc, res);
    } else {
      strb_clear(&src);
    }
    return res;
}

static void cuda_retainkernel(gpukernel *k) {
  ASSERT_KER(k);
  k->refcnt++;
}

static void cuda_freekernel(gpukernel *k) {
  ASSERT_KER(k);
  _cuda_freekernel(k);
}

static int cuda_kernelsetarg(gpukernel *k, unsigned int i, void *arg) {
  ASSERT_KER(k);
  if (i >= k->argcount)
    return error_set(k->ctx->err, GA_VALUE_ERROR, "index is beyond the last argument");
  k->args[i] = arg;
  return GA_NO_ERROR;
}

static int cuda_callkernel(gpukernel *k, unsigned int n,
                           const size_t *gs, const size_t *ls,
                           size_t shared, void **args) {
    cuda_context *ctx = k->ctx;
    unsigned int i;

    ASSERT_KER(k);
    cuda_enter(ctx);

    if (args == NULL)
      args = k->args;

    for (i = 0; i < k->argcount; i++) {
      if (k->types[i] == GA_BUFFER) {
        /* We don't have any better info for now */
        GA_CUDA_EXIT_ON_ERROR(ctx,
            cuda_wait((gpudata *)args[i], CUDA_WAIT_ALL));
      }
    }

    switch (n) {
    case 1:
      CUDA_EXIT_ON_ERROR(ctx, cuLaunchKernel(k->k, gs[0], 1, 1, ls[0], 1, 1,
                                             shared, ctx->s, args, NULL));
      break;
    case 2:
      CUDA_EXIT_ON_ERROR(ctx, cuLaunchKernel(k->k, gs[0], gs[1], 1,
                                             ls[0], ls[1], 1, shared,
                                             ctx->s, args, NULL));
      break;
    case 3:
      CUDA_EXIT_ON_ERROR(ctx, cuLaunchKernel(k->k, gs[0], gs[1], gs[2],
                                             ls[0], ls[1], ls[2], shared,
                                             ctx->s, args, NULL));
      break;
    default:
      cuda_exit(ctx);
      return error_set(ctx->err, GA_VALUE_ERROR, "Call with more than 3 dimensions");
    }

    for (i = 0; i < k->argcount; i++) {
      if (k->types[i] == GA_BUFFER) {
        /* We don't have any better info for now */
        GA_CUDA_EXIT_ON_ERROR(ctx,
            cuda_record((gpudata *)args[i], CUDA_WAIT_ALL));
      }
    }

    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_kernelbin(gpukernel *k, size_t *sz, void **obj) {
  void *res = malloc(k->bin_sz);
  if (res == NULL)
    return error_sys(k->ctx->err, "malloc");
  memcpy(res, k->bin, k->bin_sz);
  *sz = k->bin_sz;
  *obj = res;
  return GA_NO_ERROR;
}

static int cuda_sync(gpudata *b) {
  cuda_context *ctx = (cuda_context *)b->ctx;
  int err = GA_NO_ERROR;

  ASSERT_BUF(b);
  cuda_enter(ctx);
  if (ctx->flags & GA_CTX_SINGLE_STREAM) {
    CUDA_EXIT_ON_ERROR(ctx, cuStreamSynchronize(ctx->s));
  } else {
    CUDA_EXIT_ON_ERROR(ctx, cuEventSynchronize(b->wev));
    CUDA_EXIT_ON_ERROR(ctx, cuEventSynchronize(b->rev));
  }
  cuda_exit(ctx);
  return err;
}

static int cuda_transfer(gpudata *dst, size_t dstoff,
                         gpudata *src, size_t srcoff, size_t sz) {
  ASSERT_BUF(src);
  ASSERT_BUF(dst);

  /* The forced synchronization are there because they are required
     for proper inter-device correctness. */

  cuda_enter(dst->ctx);
  /* Make sure we have a rev for the source */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_records(src, CUDA_WAIT_READ|CUDA_WAIT_FORCE, src->ctx->mem_s));
  /* Make the destination stream wait for it */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_waits(src, CUDA_WAIT_READ|CUDA_WAIT_FORCE, dst->ctx->mem_s));

  /* Also wait on the destination buffer */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_waits(dst, CUDA_WAIT_WRITE, dst->ctx->mem_s));

  CUDA_EXIT_ON_ERROR(dst->ctx,
      cuMemcpyPeerAsync(dst->ptr+dstoff, dst->ctx->ctx,
                        src->ptr+srcoff, src->ctx->ctx,
                        sz, dst->ctx->mem_s));

  /* This records the event in dst->wev */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_records(dst, CUDA_WAIT_WRITE|CUDA_WAIT_FORCE, dst->ctx->mem_s));
  /* This makes the source stream wait on the wev of dst */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_waits(dst, CUDA_WAIT_WRITE|CUDA_WAIT_FORCE, src->ctx->mem_s));

  /* This records the event on src->rev */
  GA_CUDA_EXIT_ON_ERROR(dst->ctx,
      cuda_records(src, CUDA_WAIT_READ, src->ctx->mem_s));

  cuda_exit(dst->ctx);
  return GA_NO_ERROR;
}

extern gpuarray_blas_ops cublas_ops;
extern gpuarray_comm_ops nccl_ops;

static int cuda_property(gpucontext *c, gpudata *buf, gpukernel *k, int prop_id,
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

  if (prop_id < GA_BUFFER_PROP_START) {
    if (ctx == NULL)
      return error_set(global_err, GA_VALUE_ERROR,
                       "Attempting to get a context property with no context");
  } else if (prop_id < GA_KERNEL_PROP_START) {
    if (buf == NULL)
      return error_set(ctx ? ctx->err : global_err, GA_VALUE_ERROR,
                       "Attempting to get a buffer property with no buffer");
  } else {
    if (k == NULL)
      return error_set(ctx ? ctx->err : global_err, GA_VALUE_ERROR,
                       "Attempting to get a kernel property with no kernel");
  }

#define GETPROP(prop, type) do {                                   \
    cuda_enter(ctx);                                               \
    CUDA_EXIT_ON_ERROR(ctx, cuCtxGetDevice(&id));                  \
    CUDA_EXIT_ON_ERROR(ctx, cuDeviceGetAttribute(&i, (prop), id)); \
    cuda_exit(ctx);                                                \
    *((type *)res) = i;                                            \
  } while(0)

  switch (prop_id) {
    CUdevice id;
    int i;
    size_t sz;

  case GA_CTX_PROP_DEVNAME:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuCtxGetDevice(&id));
    CUDA_EXIT_ON_ERROR(ctx, cuDeviceGetName((char *)res, 256, id));
    cuda_exit(ctx);
    return GA_NO_ERROR;

  case GA_CTX_PROP_PCIBUSID:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuCtxGetDevice(&id));
    CUDA_EXIT_ON_ERROR(ctx, cuDeviceGetPCIBusId((char *)res, 13, id));
    cuda_exit(ctx);
    return GA_NO_ERROR;

  case GA_CTX_PROP_LARGEST_MEMBLOCK:
    *((size_t *)res) = largest_size(ctx);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXLSIZE:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_LMEMSIZE:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_NUMPROCS:
    GETPROP(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, unsigned int);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXGSIZE:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_BLAS_OPS:
    GA_CHECK(load_libcublas(major, minor, ctx->err));
    *((gpuarray_blas_ops **)res) = &cublas_ops;
    return GA_NO_ERROR;

  case GA_CTX_PROP_COMM_OPS:
    *((gpuarray_comm_ops**)res) = &nccl_ops;
    return GA_NO_ERROR;

  case GA_CTX_PROP_BIN_ID:
    *((const char **)res) = ctx->bin_id;
    return GA_NO_ERROR;

  case GA_CTX_PROP_ERRBUF:
    *((gpudata **)res) = ctx->errbuf;
    return GA_NO_ERROR;

  case GA_CTX_PROP_TOTAL_GMEM:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuMemGetInfo(&sz, (size_t *)res));
    cuda_exit(ctx);
    return GA_NO_ERROR;

  case GA_CTX_PROP_FREE_GMEM:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuMemGetInfo((size_t *)res, &sz));
    cuda_exit(ctx);
    return GA_NO_ERROR;

  case GA_CTX_PROP_NATIVE_FLOAT16:
    /* We claim that nobody supports this for now */
    *((int *)res) = 0;
    return CUDA_SUCCESS;

  case GA_CTX_PROP_MAXGSIZE0:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXGSIZE1:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXGSIZE2:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXLSIZE0:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXLSIZE1:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, size_t);
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXLSIZE2:
    GETPROP(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, size_t);
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_REFCNT:
    *((unsigned int *)res) = buf->refcnt;
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_SIZE:
    *((size_t *)res) = buf->sz;
    return GA_NO_ERROR;

  case GA_BUFFER_PROP_CTX:
  case GA_KERNEL_PROP_CTX:
    *((gpucontext **)res) = (gpucontext *)ctx;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_MAXLSIZE:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuFuncGetAttribute(&i, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, k->k));
    cuda_exit(ctx);
    *((size_t *)res) = i;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_PREFLSIZE:
    cuda_enter(ctx);
    CUDA_EXIT_ON_ERROR(ctx, cuCtxGetDevice(&id));
    CUDA_EXIT_ON_ERROR(ctx, cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_WARP_SIZE, id));
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
    return error_fmt(ctx->err, GA_INVALID_ERROR, "Invalid property: %d", prop_id);
  }
}

static const char *cuda_error(gpucontext *c) {
  cuda_context *ctx = (cuda_context *)c;
  const char *errstr = NULL;
  if (ctx == NULL)
    return global_err->msg;
  else
    return ctx->err->msg;
  return errstr;
}

const gpuarray_buffer_ops cuda_ops = {cuda_get_platform_count,
                                      cuda_get_device_count,
                                      cuda_init,
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
                                      cuda_kernelsetarg,
                                      cuda_callkernel,
                                      cuda_kernelbin,
                                      cuda_sync,
                                      cuda_transfer,
                                      cuda_property,
                                      cuda_error};
