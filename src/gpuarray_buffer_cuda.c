#define _CRT_SECURE_NO_WARNINGS

#include "private.h"
#include "private_cuda.h"

#include <sys/types.h>

#include <assert.h>
#include <stdlib.h>

#include <cache.h>

#include "util/strb.h"
#include "util/xxhash.h"

#include "gpuarray/buffer.h"
#include "gpuarray/util.h"
#include "gpuarray/error.h"
#include "gpuarray/extension.h"
#include "gpuarray/buffer_blas.h"

/* Allocations will be made in blocks of at least this size */
#define BLOCK_SIZE (4 * 1024 * 1024)

/* No returned allocations will be smaller than this size.
   Also, they will be aligned to this size. */
#define FRAG_SIZE (16)

static CUresult err;

static void cuda_freekernel(gpukernel *);
static int cuda_property(void *, gpudata *, gpukernel *, int, void *);
static int cuda_waits(gpudata *, int, CUstream);
static int cuda_records(gpudata *, int, CUstream);

#include "cache_extcopy.h"

static int detect_arch(const char *prefix, char *ret, CUresult *err);
static gpudata *new_gpudata(cuda_context *ctx, CUdeviceptr ptr, size_t size);

void *cuda_make_ctx(CUcontext ctx, int flags) {
  cuda_context *res;
  void *p;

  res = malloc(sizeof(*res));
  if (res == NULL)
    return NULL;
  res->ctx = ctx;
  res->err = CUDA_SUCCESS;
  res->blas_handle = NULL;
  res->refcnt = 1;
  res->flags = flags;
  res->enter = 0;
  res->freeblocks = NULL;
  if (detect_arch(ARCH_PREFIX, res->bin_id, &err)) {
    goto fail_cache;
  }
  res->extcopy_cache = cache_lru(64, 32, (cache_eq_fn)extcopy_eq,
                                 (cache_hash_fn)extcopy_hash,
                                 (cache_freek_fn)extcopy_free,
                                 (cache_freev_fn)cuda_freekernel);
  if (res->extcopy_cache == NULL) {
    goto fail_cache;
  }
  err = cuStreamCreate(&res->s, 0);
  if (err != CUDA_SUCCESS) {
    goto fail_stream;
  }
  err = cuStreamCreate(&res->mem_s, CU_STREAM_NON_BLOCKING);
  if (err != CUDA_SUCCESS) {
    goto fail_mem_stream;
  }
  err = cuMemAllocHost(&p, 16);
  if (err != CUDA_SUCCESS) {
    goto fail_errbuf;
  }
  memset(p, 0, 16);
  /* Need to tag for new_gpudata */
  TAG_CTX(res);
  res->errbuf = new_gpudata(res, (CUdeviceptr)p, 16);
  if (res->errbuf == NULL) {
    err = res->err;
    goto fail_end;
  }
  res->errbuf->flags |= CUDA_MAPPED_PTR;
  return res;
 fail_end:
  cuMemFreeHost(p);
 fail_errbuf:
  cuStreamDestroy(res->mem_s);
 fail_mem_stream:
  cuStreamDestroy(res->s);
 fail_stream:
  cache_destroy(res->extcopy_cache);
 fail_cache:
  free(res);
  return NULL;
}

static void deallocate(gpudata *);

static void cuda_free_ctx(cuda_context *ctx) {
  gpuarray_blas_ops *blas_ops;
  gpudata *next, *curr;
#if CUDA_VERSION >= 7000
  CUdevice dev;
#endif

  ASSERT_CTX(ctx);
  ctx->refcnt--;
  if (ctx->refcnt == 0) {
    assert(ctx->enter == 0 && "Context was active when freed!");
    if (ctx->blas_handle != NULL) {
      ctx->err = cuda_property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS,
                               &blas_ops);
      blas_ops->teardown(ctx);
    }
    cuMemFreeHost((void *)ctx->errbuf->ptr);
    deallocate(ctx->errbuf);

    cuStreamDestroy(ctx->s);

    /* Clear out the freelist */
    for (curr = ctx->freeblocks; curr != NULL; curr = next) {
      next = curr->next;
      cuMemFree(curr->ptr);
      deallocate(curr);
    }

    if (!(ctx->flags & DONTFREE)) {
#if CUDA_VERSION < 7000
      cuCtxDestroy(ctx->ctx);
#else
      cuCtxPushCurrent(ctx->ctx);
      cuCtxGetDevice(&dev);
      cuCtxPopCurrent(NULL);
      cuDevicePrimaryCtxRelease(dev);
#endif
    }
    cache_destroy(ctx->extcopy_cache);
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
  int fl = CU_EVENT_DISABLE_TIMING;

  res = malloc(sizeof(*res));
  if (res == NULL) return NULL;

  res->refcnt = 0;
  res->sz = size;

  res->flags = 0;

  cuda_enter(ctx);

  if (ctx->flags & GA_CTX_MULTI_THREAD)
    fl |= CU_EVENT_BLOCKING_SYNC;
  ctx->err = cuEventCreate(&res->rev, fl);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    free(res);
    return NULL;
  }

  ctx->err = cuEventCreate(&res->wev, fl);
  if (ctx->err != CUDA_SUCCESS) {
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

gpudata *cuda_make_buf(void *c, CUdeviceptr p, size_t sz) {
  cuda_context *ctx = (cuda_context *)c;
  gpudata *res = new_gpudata(ctx, p, sz);

  if (res == NULL) return NULL;

  res->refcnt = 1;
  res->flags |= DONTFREE;
  res->ctx->refcnt++;

  return res;
}

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
    "#ifdef NAN\n"
    "#undef NAN\n"
    "#endif\n"
    "#define NAN __int_as_float(0x7fffffff)\n"
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
    "#define store_half(p, v) (*(p) = __float2half_rn(v))\n";

/* XXX: add complex, quads, longlong */
/* XXX: add vector types */

static void *do_init(CUdevice dev, int flags, int *ret) {
    cuda_context *res;
    CUcontext ctx;
    unsigned int fl = CU_CTX_SCHED_AUTO;
#if CUDA_VERSION >= 7000
    unsigned int cur_fl;
    int act;
#endif
    int i;

    CHKFAIL(NULL);
    if (flags & GA_CTX_SINGLE_THREAD)
      fl = CU_CTX_SCHED_SPIN;
    if (flags & GA_CTX_MULTI_THREAD)
      fl = CU_CTX_SCHED_YIELD;
    err = cuDeviceGetAttribute(&i, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev);
    CHKFAIL(NULL);
    if (i != 1)
      FAIL(NULL, GA_UNSUPPORTED_ERROR);
#if CUDA_VERSION < 7000
    err = cuCtxCreate(&ctx, fl, dev);
    CHKFAIL(NULL);
#else
    err = cuDevicePrimaryCtxGetState(dev, &cur_fl, &act);
    CHKFAIL(NULL);
    if (act == 1) {
      if ((cur_fl & fl) != fl)
        FAIL(NULL, GA_INVALID_ERROR);
    } else {
      err = cuDevicePrimaryCtxSetFlags(dev, fl);
      CHKFAIL(NULL);
    }
    err = cuDevicePrimaryCtxRetain(&ctx, dev);
    CHKFAIL(NULL);
    err = cuCtxPushCurrent(ctx);
    CHKFAIL(NULL);
#endif
    res = cuda_make_ctx(ctx, 0);
    if (res == NULL) {
#if CUDA_VERSION < 7000
      cuCtxDestroy(ctx);
#else
      cuDevicePrimaryCtxRelease(dev);
#endif
      FAIL(NULL, GA_IMPL_ERROR);
    }
    res->flags |= flags;
    /* Don't leave the context on the thread stack */
    cuCtxPopCurrent(NULL);

    return res;
}
static void *cuda_init(int ord, int flags, int *ret) {
    CUdevice dev;
    cuda_context *res;
    static int init_done = 0;

    if (!init_done) {
      err = cuInit(0);
      CHKFAIL(NULL);
      init_done = 1;
    }

    if (ord == -1) {
      int i, c;
      err = cuDeviceGetCount(&c);
      CHKFAIL(NULL);
      for (i = 0; i < c; i++) {
        err = cuDeviceGet(&dev, i);
        CHKFAIL(NULL);
        res = do_init(dev, flags, NULL);
        if (res != NULL)
          return res;
      }
      FAIL(NULL, GA_NODEV_ERROR);
    } else {
      err = cuDeviceGet(&dev, ord);
      CHKFAIL(NULL);
      return do_init(dev, flags, ret);
    }
}
static void cuda_deinit(void *c) {
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

/*
 * Allocate a new block and place in on the freelist. Will allocate
 * the bigger of the requested size and BLOCK_SIZE to avoid allocating
 * multiple small blocks.
 */
static int allocate(cuda_context *ctx, gpudata **res, gpudata **prev,
                    size_t size) {
  CUdeviceptr ptr;
  gpudata *next;
  *prev = NULL;

  if (!(ctx->flags & GA_CTX_DISABLE_ALLOCATION_CACHE))
    if (size < BLOCK_SIZE) size = BLOCK_SIZE;

  cuda_enter(ctx);

  ctx->err = cuMemAlloc(&ptr, size);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_exit(ctx);
    return GA_IMPL_ERROR;
  }

  *res = new_gpudata(ctx, ptr, size);

  cuda_exit(ctx);

  if (*res == NULL) {
    cuMemFree(ptr);
    return GA_MEMORY_ERROR;
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
      return GA_MEMORY_ERROR;
    /* Make sure the chain keeps going */
    split->next = curr->next;
    curr->next = NULL;
    /* Make sure we don't start using the split buffer too soon */
    cuda_wait(curr, CUDA_WAIT_ALL);
    cuda_record(split, CUDA_WAIT_ALL);
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

static gpudata *cuda_alloc(void *c, size_t size, void *data, int flags,
			   int *ret) {
  gpudata *res = NULL, *prev = NULL;
  cuda_context *ctx = (cuda_context *)c;
  size_t asize;
  int err;

  if ((flags & GA_BUFFER_INIT) && data == NULL) FAIL(NULL, GA_VALUE_ERROR);
  if ((flags & (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) ==
      (GA_BUFFER_READ_ONLY|GA_BUFFER_WRITE_ONLY)) FAIL(NULL, GA_VALUE_ERROR);

  /* TODO: figure out how to make this work */
  if (flags & GA_BUFFER_HOST) FAIL(NULL, GA_DEVSUP_ERROR);

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

  if (res == NULL) {
    err = allocate(ctx, &res, &prev, asize);
    if (err != GA_NO_ERROR)
      FAIL(NULL, err);
  }

  err = extract(res, prev, asize);
  if (err != GA_NO_ERROR)
    FAIL(NULL, err);
  /* It's out of the freelist, so add a ref */
  res->ctx->refcnt++;
  /* We consider this buffer allocated and ready to go */
  res->refcnt = 1;

  if (flags & GA_BUFFER_INIT) {
    err = cuda_write(res, 0, data, size);
    if (err != GA_NO_ERROR) {
      cuda_free(res);
      FAIL(NULL, err);
    }
  }

  return res;
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
        cuda_wait(d, CUDA_WAIT_ALL);
        cuda_record(prev, CUDA_WAIT_ALL);
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

static int cuda_share(gpudata *a, gpudata *b, int *ret) {
  ASSERT_BUF(a);
  ASSERT_BUF(b);
  return (a->ctx == b->ctx && a->sz != 0 && b->sz != 0 &&
          ((a->ptr <= b->ptr && a->ptr + a->sz > b->ptr) ||
           (b->ptr <= a->ptr && b->ptr + b->sz > a->ptr)));
}

static int cuda_waits(gpudata *a, int flags, CUstream s) {
  ASSERT_BUF(a);
  /* If others are only reads, no need to wait */
  cuda_enter(a->ctx);
  if (flags & CUDA_WAIT_READ) {
    /* We wait for writes that happened before since multiple reads at
     * the same time are fine */
    a->ctx->err = cuStreamWaitEvent(s, a->wev, 0);
    if (a->ctx->err != CUDA_SUCCESS) {
      cuda_exit(a->ctx);
      return GA_IMPL_ERROR;
    }
  }
  if (flags & CUDA_WAIT_WRITE) {
    /* Make sure to not disturb previous reads */
    a->ctx->err = cuStreamWaitEvent(s, a->rev, 0);
    if (a->ctx->err != CUDA_SUCCESS) {
      cuda_exit(a->ctx);
      return GA_IMPL_ERROR;
    }
  }
  cuda_exit(a->ctx);
  return GA_NO_ERROR;
}

int cuda_wait(gpudata *a, int flags) {
  return cuda_waits(a, flags, a->ctx->s);
}

static int cuda_records(gpudata *a, int flags, CUstream s) {
  ASSERT_BUF(a);
  cuda_enter(a->ctx);
  if (flags & CUDA_WAIT_READ)
    a->ctx->err = cuEventRecord(a->rev, s);
  if (flags & CUDA_WAIT_WRITE)
    a->ctx->err = cuEventRecord(a->wev, s);
  cuda_exit(a->ctx);
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
    if (src->ctx != dst->ctx) return GA_VALUE_ERROR;

    if (sz == 0) return GA_NO_ERROR;

    if ((dst->sz - dstoff) < sz || (src->sz - srcoff) < sz)
        return GA_VALUE_ERROR;

    cuda_enter(ctx);

    cuda_wait(src, CUDA_WAIT_READ);
    cuda_wait(dst, CUDA_WAIT_WRITE);

    ctx->err = cuMemcpyDtoDAsync(dst->ptr + dstoff, src->ptr + srcoff, sz,
                                 ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    cuda_record(src, CUDA_WAIT_READ);
    cuda_record(dst, CUDA_WAIT_WRITE);
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

    if (src->flags & CUDA_MAPPED_PTR) {
      ctx->err = cuEventSynchronize(src->wev);
      if (ctx->err != CUDA_SUCCESS) {
        cuda_exit(ctx);
        return GA_IMPL_ERROR;
      }
      memcpy(dst, (void *)(src->ptr + srcoff), sz);
    } else {
      cuda_waits(src, CUDA_WAIT_READ, ctx->mem_s);

      ctx->err = cuMemcpyDtoHAsync(dst, src->ptr + srcoff, sz, ctx->mem_s);
      if (ctx->err != CUDA_SUCCESS) {
        cuda_exit(ctx);
        return GA_IMPL_ERROR;
      }
      cuda_records(src, CUDA_WAIT_READ, ctx->mem_s);
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

    if (dst->flags & CUDA_MAPPED_PTR) {
      ctx->err = cuEventSynchronize(dst->rev);
      if (ctx->err != CUDA_SUCCESS) {
        cuda_exit(ctx);
        return GA_IMPL_ERROR;
      }
      memcpy((void *)(dst->ptr + dstoff), src, sz);
    } else {
      cuda_waits(dst, CUDA_WAIT_WRITE, ctx->mem_s);

      ctx->err = cuMemcpyHtoDAsync(dst->ptr + dstoff, src, sz, ctx->mem_s);
      if (ctx->err != CUDA_SUCCESS) {
        cuda_exit(ctx);
        return GA_IMPL_ERROR;
      }

      cuda_records(dst, CUDA_WAIT_WRITE, ctx->mem_s);
    }
    cuda_exit(ctx);
    return GA_NO_ERROR;
}

static int cuda_memset(gpudata *dst, size_t dstoff, int data) {
    cuda_context *ctx = dst->ctx;

    ASSERT_BUF(dst);

    if ((dst->sz - dstoff) == 0) return GA_NO_ERROR;

    cuda_enter(ctx);

    cuda_wait(dst, CUDA_WAIT_WRITE);

    ctx->err = cuMemsetD8Async(dst->ptr + dstoff, data, dst->sz - dstoff,
                               ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    cuda_record(dst, CUDA_WAIT_WRITE);
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

static int detect_arch(const char *prefix, char *ret, CUresult *err) {
  CUdevice dev;
  int major, minor;
  int res;
  size_t sz = strlen(prefix) + 3;
  *err = cuCtxGetDevice(&dev);
  if (*err != CUDA_SUCCESS) return GA_IMPL_ERROR;
  *err = get_cc(dev, &major, &minor);
  if (*err != CUDA_SUCCESS) return GA_IMPL_ERROR;
  res = snprintf(ret, sz, "%s%d%d", prefix, major, minor);
  if (res == -1 || res > sz) return GA_UNSUPPORTED_ERROR;
  return GA_NO_ERROR;
}

static cache *compile_cache = NULL;

typedef struct _srckey {
  const char *src;
  size_t len;
  char arch[BIN_ID_LEN];
} srckey;

static void src_free(void *_k) {
  srckey *k = (srckey *)_k;
  free((void *)k->src);
  free(k);
}

static int src_eq(void *_k1, void *_k2) {
  srckey *k1 = (srckey *)_k1;
  srckey *k2 = (srckey *)_k2;
  return (k1->len == k2->len &&
          strcmp(k1->arch, k2->arch) == 0 &&
          memcmp(k1->src, k2->src, k1->len) == 0);
}

static uint32_t src_hash(void *_k) {
  srckey *k = (srckey *)_k;
  XXH32_state_t h;
  /* seed is an arbitrary, but fixed value */
  XXH32_reset(&h, 42);
  XXH32_update(&h, (void *)k->src, k->len);
  XXH32_update(&h, (void *)k->arch, sizeof(k->arch));
  return XXH32_digest(&h);
}

typedef struct _binval {
  void *bin;
  size_t len;
} binval;

static void bin_free(void *_v) {
  binval *v = (binval *)_v;
  free(v->bin);
  free(v);
}

#ifdef WITH_NVRTC

#include <nvrtc.h>

static void *call_compiler(const char *src, size_t len, const char *arch_arg,
                           size_t *bin_len, char **log, size_t *log_len,
                           int *ret) {
  nvrtcProgram prog;
  void *buf = NULL;
  size_t buflen;
  const char *opts[4] = {
    "-arch", ""
    , "-G", "-lineinfo"
  };
  nvrtcResult err, err2;

  opts[1] = arch_arg;

  err = nvrtcCreateProgram(&prog, src, NULL, 0, NULL, NULL);
  if (err != NVRTC_SUCCESS) FAIL(NULL, GA_SYS_ERROR);

  err = nvrtcCompileProgram(prog,
#ifdef DEBUG
                            4,
#else
                            2,
#endif
                            opts);
  if (log != NULL) {
    err2 = nvrtcGetProgramLogSize(prog, &buflen);
    if (err2 != NVRTC_SUCCESS) goto end2;
    buf = malloc(buflen);
    if (buf == NULL) goto end2;
    err2 = nvrtcGetProgramLog(prog, (char *)buf);
    if (err2 != NVRTC_SUCCESS) goto end2;
    if (log_len != NULL) *log_len = buflen;
    *log = (char *)buf;
    buf = NULL;
  }
end2:
  if (err != NVRTC_SUCCESS) goto end;

  err = nvrtcGetPTXSize(prog, &buflen);
  if (err != NVRTC_SUCCESS) goto end;

  buf = malloc(buflen);
  if (buf == NULL) {
    nvrtcDestroyProgram(&prog);
    FAIL(NULL, GA_MEMORY_ERROR);
  }

  err = nvrtcGetPTX(prog, (char *)buf);
  if (err != NVRTC_SUCCESS) goto end;

  *bin_len = buflen;

end:
  nvrtcDestroyProgram(&prog);
  if (err != NVRTC_SUCCESS) {
    free(buf);
    FAIL(NULL, GA_SYS_ERROR);
  }
  return buf;
}

#else /* WITH_NVRTC */

#include <sys/stat.h>

#include <fcntl.h>
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
#define open _open
#else
#include <unistd.h>
#endif

static const char *TMP_VAR_NAMES[] = {"GPUARRAY_TMPDIR", "TMPDIR", "TMP",
                                      "TEMP", "USERPROFILE"};


static void *call_compiler(const char *src, size_t len, const char *arch_arg,
                           size_t *bin_len, char **log, size_t *log_len,
                           int *ret) {
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

    /* Don't want to write the final NUL */
    s = write(fd, src, len-1);
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
#ifdef DEBUG
      /* Only cleanup if GPUARRAY_NOCLEANUP is not set */
      if (getenv("GPUARRAY_NOCLEANUP") == NULL)
#endif
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
    /* fd is blocking; should have complete read */
    if (s == -1) {
      free(buf);
      FAIL(NULL, GA_SYS_ERROR);
    }

    *bin_len = (size_t)st.st_size;
    return buf;
}

#endif /* WITH_NVRTC */

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

static gpukernel *cuda_newkernel(void *c, unsigned int count,
                                 const char **strings, const size_t *lengths,
                                 const char *fname, unsigned int argcount,
                                 const int *types, int flags, int *ret,
                                 char **err_str) {
    cuda_context *ctx = (cuda_context *)c;
    strb sb = STRB_STATIC_INIT;
    char *bin, *log = NULL;
    srckey k, *ak;
    binval *av;
    gpukernel *res;
    size_t bin_len = 0, log_len = 0;
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
      bin = memdup(strings[0], lengths[0]);
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

      strb_append0(&sb);

      if (strb_error(&sb)) {
        strb_clear(&sb);
        cuda_exit(ctx);
        return NULL;
      }

      if (ptx_mode) {
        bin = sb.s;
        bin_len = sb.l;
      } else {
        bin = NULL;
        if (compile_cache != NULL) {
          k.src = sb.s;
          k.len = sb.l;
          memcpy(k.arch, ctx->bin_id, BIN_ID_LEN);
          av = cache_get(compile_cache, &k);
          if (av != NULL) {
            bin = memdup(av->bin, av->len);
            bin_len = av->len;
          }
        }
        if (bin == NULL) {
          bin = call_compiler(sb.s, sb.l, ctx->bin_id, &bin_len,
                              &log, &log_len, ret);
        }
        if (bin == NULL) {
          if (err_str != NULL) {
            strb debug_msg = STRB_STATIC_INIT;

            // We're substituting debug_msg for a string with this first line:
            strb_appends(&debug_msg, "CUDA kernel build failure ::\n");

            /* Delete the final NUL */
            sb.l--;
            gpukernel_source_with_line_numbers(1, (const char **)&sb.s,
                                               &sb.l, &debug_msg);

            if (log != NULL) {
              strb_appends(&debug_msg, "\nCompiler log:\n");
              strb_appendn(&debug_msg, log, log_len);
              free(log);
            }
            *err_str = strb_cstr(&debug_msg);
            // *err_str will be free()d by the caller (see docs in kernel.h)
          }
          strb_clear(&sb);
          cuda_exit(ctx);
          return NULL;
        }
        if (compile_cache == NULL)
          compile_cache = cache_twoq(16, 16, 16, 8, src_eq, src_hash, src_free,
                                     bin_free);

        if (compile_cache != NULL) {
          ak = malloc(sizeof(*ak));
          av = malloc(sizeof(*av));
          if (ak == NULL || av == NULL) {
            free(ak);
            free(av);
            goto done;
          }
          ak->src = memdup(sb.s, sb.l);
          if (ak->src == NULL) {
            free(ak);
            free(av);
            goto done;
          }
          ak->len = sb.l;
          memmove(ak->arch, ctx->bin_id, BIN_ID_LEN);
          av->len = bin_len;
          av->bin = memdup(bin, bin_len);
          if (av->bin == NULL) {
            src_free(ak);
            free(av);
            goto done;
          }
          cache_add(compile_cache, ak, av);
        }
      done:
        strb_clear(&sb);
      }
    }

    res = calloc(1, sizeof(*res));
    if (res == NULL) {
      free(bin);
      cuda_exit(ctx);
      FAIL(NULL, GA_SYS_ERROR);
    }

    res->bin_sz = bin_len;
    res->bin = bin;

    res->refcnt = 1;
    res->argcount = argcount;
    res->types = calloc(argcount, sizeof(int));
    if (res->types == NULL) {
      _cuda_freekernel(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_MEMORY_ERROR);
    }
    memcpy(res->types, types, argcount*sizeof(int));
    res->args = calloc(argcount, sizeof(void *));
    if (res->args == NULL) {
      _cuda_freekernel(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_MEMORY_ERROR);
    }

    ctx->err = cuModuleLoadData(&res->m, bin);

    if (ctx->err != CUDA_SUCCESS) {
      _cuda_freekernel(res);
      cuda_exit(ctx);
      FAIL(NULL, GA_IMPL_ERROR);
    }

    ctx->err = cuModuleGetFunction(&res->k, res->m, fname);
    if (ctx->err != CUDA_SUCCESS) {
      _cuda_freekernel(res);
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
  _cuda_freekernel(k);
}

static int cuda_kernelsetarg(gpukernel *k, unsigned int i, void *arg) {
  if (i >= k->argcount)
    return GA_VALUE_ERROR;
  k->args[i] = arg;
  return GA_NO_ERROR;
}

static int cuda_callkernel(gpukernel *k, unsigned int n,
                           const size_t *bs, const size_t *gs,
                           size_t shared, void **args) {
    cuda_context *ctx = k->ctx;
    unsigned int i;

    ASSERT_KER(k);
    cuda_enter(ctx);

    for (i = 0; i < k->argcount; i++) {
      if (k->types[i] == GA_BUFFER) {
	/* We don't have any better info for now */
	cuda_wait((gpudata *)args[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      }
    }

    if (args == NULL)
      args = k->args;

    switch (n) {
    case 1:
      ctx->err = cuLaunchKernel(k->k, gs[0], 1, 1, bs[0], 1, 1, shared,
                                ctx->s, args, NULL);
      break;
    case 2:
      ctx->err = cuLaunchKernel(k->k, gs[0], gs[1], 1, bs[0], bs[1], 1, shared,
                                ctx->s, args, NULL);
      break;
    case 3:
      ctx->err = cuLaunchKernel(k->k, gs[0], gs[1], gs[2], bs[0], bs[1], bs[2],
                                shared, ctx->s, args, NULL);
      break;
    default:
      cuda_exit(ctx);
      return GA_VALUE_ERROR;
    }
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      return GA_IMPL_ERROR;
    }

    for (i = 0; i < k->argcount; i++) {
      if (k->types[i] == GA_BUFFER) {
	/* We don't have any better info for now */
	cuda_record((gpudata *)args[i], CUDA_WAIT_READ|CUDA_WAIT_WRITE);
      }
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
  int err = GA_NO_ERROR;

  ASSERT_BUF(b);
  cuda_enter(ctx);
  ctx->err = cuEventSynchronize(b->wev);
  if (ctx->err != CUDA_SUCCESS)
    err = GA_IMPL_ERROR;
  ctx->err = cuEventSynchronize(b->rev);
  if (ctx->err != CUDA_SUCCESS)
    err = GA_IMPL_ERROR;
  cuda_exit(ctx);
  return err;
}

static const char ELEM_HEADER_PTX[] = ".version %s\n.target %s\n\n"
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

static inline int gen_extcopy_kernel(const extcopy_args *a,
				     cuda_context *ctx, gpukernel **v,
				     size_t nEls) {
  strb sb = STRB_STATIC_INIT;
  int res = GA_SYS_ERROR;
  int flags = GA_USE_PTX;
  unsigned int bits = sizeof(void *)*8;
  int types[2];
  const char *in_t, *in_ld_t;
  const char *out_t, *out_ld_t;
  const char *rmod;

  in_t = map_t(a->itype);
  out_t = map_t(a->otype);
  /* Since float16 ('f16') is not a fully-supported type we need to use
     it as b16 (basically uint16) for read and write operations. */
  if (a->itype == GA_HALF)
    in_ld_t = "b16";
  else
    in_ld_t = in_t;
  if (a->otype == GA_HALF)
    out_ld_t = "b16";
  else
    out_ld_t = out_t;
  rmod = get_rmod(a->itype, a->otype);
  if (in_t == NULL || out_t == NULL) return GA_DEVSUP_ERROR;

  strb_appendf(&sb, ELEM_HEADER_PTX, "4.1", ctx->bin_id,
               bits, bits, bits, bits, in_t, out_t, bits,
               bits, bits, bits, bits, nEls, bits, bits);

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
	       in_ld_t, a->ioff,
	       rmod, out_t, in_t,
	       bits,
	       bits, bits,
	       bits,
	       out_ld_t, a->ooff);

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

static int cuda_extcopy(gpudata *input, size_t ioff, gpudata *output,
                        size_t ooff, int intype, int outtype,
                        unsigned int a_nd, const size_t *a_dims,
                        const ssize_t *a_str, unsigned int b_nd,
                        const size_t *b_dims, const ssize_t *b_str) {
  cuda_context *ctx = input->ctx;
  void *args[2];
  int res = GA_SYS_ERROR;
  unsigned int i;
  size_t nEls = 1, ls, gs;
  gpukernel *k;
  extcopy_args a, *aa;

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

  k = cache_get(ctx->extcopy_cache, &a);
  if (k == NULL) {
    res = gen_extcopy_kernel(&a, input->ctx, &k, nEls);
    if (res != GA_NO_ERROR)
      return res;

    /* Cache the kernel */
    aa = memdup(&a, sizeof(a));
    if (aa == NULL) goto done;
    aa->idims = memdup(a_dims, a_nd*sizeof(size_t));
    aa->odims = memdup(b_dims, b_nd*sizeof(size_t));
    aa->istr = memdup(a_str, a_nd*sizeof(ssize_t));
    aa->ostr = memdup(b_str, b_nd*sizeof(ssize_t));
    if (aa->idims == NULL || aa->odims == NULL ||
        aa->istr == NULL || aa->ostr == NULL) {
      extcopy_free(aa);
      goto done;
    }
    /* One ref is given to the cache, we manage the other */
    cuda_retainkernel(k);
    cache_add(ctx->extcopy_cache, aa, k);
  } else {
    /* This is our reference */
    cuda_retainkernel(k);
  }
done:

  /* Cheap kernel scheduling */
  res = cuda_property(NULL, NULL, k, GA_KERNEL_PROP_MAXLSIZE, &ls);
  if (res != GA_NO_ERROR) goto fail;

  gs = ((nEls-1) / ls) + 1;
  args[0] = input;
  args[1] = output;
  res = cuda_callkernel(k, 1, &ls, &gs, 0, args);

fail:
  /* We free our reference here */
  cuda_freekernel(k);
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
    dst = cuda_alloc(ctx, sz, NULL, 0, NULL);
    if (dst == NULL) return NULL;
    cuda_enter(ctx);

    cuda_wait(src, CUDA_WAIT_READ);
    cuda_wait(dst, CUDA_WAIT_WRITE);

    ctx->err = cuMemcpyDtoDAsync(dst->ptr, src->ptr+offset, sz, ctx->s);
    if (ctx->err != CUDA_SUCCESS) {
      cuda_exit(ctx);
      cuda_free(dst);
      return NULL;
    }
    cuda_record(src, CUDA_WAIT_READ);
    cuda_record(dst, CUDA_WAIT_WRITE);

    cuda_exit(ctx);
    return dst;
  }

  dst = cuda_alloc(dst_ctx, sz, NULL, 0, NULL);
  if (dst == NULL)
    return NULL;
  cuda_enter(ctx);
  cuda_waits(src, CUDA_WAIT_READ, dst_ctx->mem_s);
  cuda_waits(dst, CUDA_WAIT_WRITE, dst_ctx->mem_s);
  ctx->err = cuMemcpyPeerAsync(dst->ptr, dst->ctx->ctx, src->ptr+offset,
			       src->ctx->ctx, sz, dst_ctx->mem_s);
  if (ctx->err != CUDA_SUCCESS) {
    cuda_free(dst);
    cuda_exit(ctx);
    return NULL;
  }

  cuda_records(dst, CUDA_WAIT_WRITE, dst_ctx->mem_s);
  cuda_records(src, CUDA_WAIT_READ, dst_ctx->mem_s);

  cuda_exit(ctx);
  return dst;
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
    size_t sz;

  case GA_CTX_PROP_DEVNAME:
    cuda_enter(ctx);
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

  case GA_CTX_PROP_ERRBUF:
    *((gpudata **)res) = ctx->errbuf;
    return GA_NO_ERROR;

  case GA_CTX_PROP_TOTAL_GMEM:
    cuda_enter(ctx);
    ctx->err = cuMemGetInfo(&sz, (size_t *)res);
    cuda_exit(ctx);
    return ctx->err == CUDA_SUCCESS ? GA_NO_ERROR : GA_IMPL_ERROR;

  case GA_CTX_PROP_FREE_GMEM:
    cuda_enter(ctx);
    ctx->err = cuMemGetInfo((size_t *)res, &sz);
    cuda_exit(ctx);
    return ctx->err == CUDA_SUCCESS ? GA_NO_ERROR : GA_IMPL_ERROR;

  case GA_CTX_PROP_NATIVE_FLOAT16:
    /* We claim that nobody supports this for now */
    *((int *)res) = 0;
    return CUDA_SUCCESS;

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
  const char *errstr = NULL;
  if (ctx == NULL)
    cuGetErrorString(err, &errstr);
  else
    cuGetErrorString(ctx->err, &errstr);
  return errstr;
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
                                      cuda_kernelsetarg,
                                      cuda_callkernel,
                                      cuda_kernelbin,
                                      cuda_sync,
                                      cuda_extcopy,
                                      cuda_transfer,
                                      cuda_property,
                                      cuda_error};
