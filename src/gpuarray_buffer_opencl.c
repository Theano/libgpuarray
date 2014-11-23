#define _CRT_SECURE_NO_WARNINGS

#include "private.h"
#include "private_opencl.h"
#include "gpuarray/buffer.h"
#include "gpuarray/util.h"
#include "gpuarray/error.h"
#include "gpuarray/buffer_blas.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#ifdef _MSC_VER
#define strdup _strdup
#endif

#define SSIZE_MIN (-(SSIZE_MAX-1))

static cl_int err;

#define FAIL(v, e) { if (ret) *ret = e; return v; }
#define CHKFAIL(v) if (err != CL_SUCCESS) FAIL(v, GA_IMPL_ERROR)

static int cl_property(void *c, gpudata *b, gpukernel *k, int p, void *r);

static cl_device_id get_dev(cl_context ctx, int *ret) {
  size_t sz;
  cl_device_id res;
  cl_device_id *ids;
  cl_int err;

  err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz);
  CHKFAIL(NULL);

  ids = malloc(sz);
  if (ids == NULL) FAIL(NULL, GA_MEMORY_ERROR);

  err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sz, ids, NULL);
  res = ids[0];
  free(ids);
  CHKFAIL(NULL);
  return res;
}

cl_ctx *cl_make_ctx(cl_context ctx) {
  cl_ctx *res;
  cl_device_id id;
  cl_command_queue_properties qprop;
  char vendor[32];
  char driver_version[16];
  cl_uint vendor_id;
  size_t len;

  id = get_dev(ctx, NULL);
  if (id == NULL) return NULL;
  err = clGetDeviceInfo(id, CL_DEVICE_QUEUE_PROPERTIES, sizeof(qprop),
                        &qprop, NULL);
  if (err != CL_SUCCESS) return NULL;

  err = clGetDeviceInfo(id, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
  if (err != CL_SUCCESS)
    return NULL;
  err = clGetDeviceInfo(id, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id,
                        NULL);
  if (err != CL_SUCCESS)
    return NULL;
  err = clGetDeviceInfo(id, CL_DRIVER_VERSION, sizeof(driver_version),
                        driver_version, NULL);
  if (err != CL_SUCCESS)
    return NULL;

  res = malloc(sizeof(*res));
  if (res == NULL) return NULL;

  res->ctx = ctx;
  res->err = CL_SUCCESS;
  res->refcnt = 1;
  res->exts = NULL;
  res->blas_handle = NULL;
  res->q = clCreateCommandQueue(ctx, id,
				qprop&CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
				&err);
  if (res->q == NULL) {
    free(res);
    return NULL;
  }

  /* Can't overflow (source is 32 + 16 + 12 and buffer is 64) */
  len = strlcpy(res->bin_id, vendor, sizeof(res->bin_id));
  snprintf(res->bin_id + len, sizeof(res->bin_id) - len, " %#x ", vendor_id);
  strlcat(res->bin_id, driver_version, sizeof(res->bin_id));

  clRetainContext(res->ctx);
  TAG_CTX(res);
  return res;
}

cl_context cl_get_ctx(void *ctx) {
  ASSERT_CTX((cl_ctx *)ctx);
  return ((cl_ctx *)ctx)->ctx;
}

cl_command_queue cl_get_stream(void *ctx) {
  ASSERT_CTX((cl_ctx *)ctx);
  return ((cl_ctx *)ctx)->q;
}

static void cl_free_ctx(cl_ctx *ctx) {
  gpuarray_blas_ops *blas_ops;

  ASSERT_CTX(ctx);
  assert(ctx->refcnt != 0);
  ctx->refcnt--;
  if (ctx->refcnt == 0) {
    CLEAR(ctx);
    if (ctx->blas_handle != NULL) {
      ctx->err = cl_property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas_ops);
      blas_ops->teardown(ctx);
    }
    clReleaseCommandQueue(ctx->q);
    clReleaseContext(ctx->ctx);
    free(ctx);
  }
}

gpudata *cl_make_buf(void *c, cl_mem buf) {
  cl_ctx *ctx = (cl_ctx *)c;
  gpudata *res;
  cl_context buf_ctx;

  ASSERT_CTX(ctx);
  ctx->err = clGetMemObjectInfo(buf, CL_MEM_CONTEXT, sizeof(buf_ctx),
                                &buf_ctx, NULL);
  if (ctx->err != CL_SUCCESS) return NULL;
  if (buf_ctx != ctx->ctx) return NULL;

  res = malloc(sizeof(*res));
  if (res == NULL) return NULL;

  res->buf = buf;
  res->ev = NULL;
  res->refcnt = 1;
  ctx->err = clRetainMemObject(buf);
  if (ctx->err != CL_SUCCESS) {
    free(res);
    return NULL;
  }
  res->ctx = ctx;
  res->ctx->refcnt++;

  TAG_BUF(res);
  return res;
}

cl_mem cl_get_buf(gpudata *g) { ASSERT_BUF(g); return g->buf; }

#define PRAGMA "#pragma OPENCL EXTENSION "
#define ENABLE " : enable\n"
#define CL_SMALL "cl_khr_byte_addressable_store"
#define CL_DOUBLE "cl_khr_fp64"
#define CL_HALF "cl_khr_fp16"

static gpukernel *cl_newkernel(void *ctx, unsigned int count,
                               const char **strings, const size_t *lengths,
                               const char *fname, unsigned int argcount,
                               const int *types, int flags, int *ret, char **err_str);
static void cl_releasekernel(gpukernel *k);
static int cl_callkernel(gpukernel *k, size_t bs[2], size_t gs[2], void **args);

static const char CL_PREAMBLE[] =
  "#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE)\n"
  "#define WITHIN_KERNEL /* empty */\n"
  "#define KERNEL __kernel\n"
  "#define GLOBAL_MEM __global\n"
  "#define LOCAL_MEM __local\n"
  "#define LOCAL_MEM_ARG __local\n"
  "#define REQD_WG_SIZE(x, y, z) __attribute__((reqd_work_group_size(x, y, z)))\n"
  "#define LID_0 get_local_id(0)\n"
  "#define LID_1 get_local_id(1)\n"
  "#define LID_2 get_local_id(2)\n"
  "#define LDIM_0 get_local_size(0)\n"
  "#define LDIM_1 get_local_size(1)\n"
  "#define LDIM_2 get_local_size(2)\n"
  "#define GID_0 get_group_id(0)\n"
  "#define GID_1 get_group_id(1)\n"
  "#define GID_2 get_group_id(2)\n"
  "#define GDIM_0 get_num_groups(0)\n"
  "#define GDIM_1 get_num_groups(1)\n"
  "#define GDIM_2 get_num_groups(2)\n"
  "#define ga_bool uchar\n"
  "#define ga_byte char\n"
  "#define ga_ubyte uchar\n"
  "#define ga_short short\n"
  "#define ga_ushort ushort\n"
  "#define ga_int int\n"
  "#define ga_uint uint\n"
  "#define ga_long long\n"
  "#define ga_ulong ulong\n"
  "#define ga_float float\n"
  "#define ga_double double\n"
  "#define ga_half half\n"
  "#define ga_size ulong\n";
/* XXX: add complex types, quad types, and longlong */
/* XXX: add vector types */

static const char *get_error_string(cl_int err) {
  /* OpenCL 1.0 error codes */
  switch (err) {
  case CL_SUCCESS:                        return "Success!";
  case CL_DEVICE_NOT_FOUND:               return "Device not found.";
  case CL_DEVICE_NOT_AVAILABLE:           return "Device not available";
  case CL_COMPILER_NOT_AVAILABLE:         return "Compiler not available";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:  return "Memory object allocation failure";
  case CL_OUT_OF_RESOURCES:               return "Out of resources";
  case CL_OUT_OF_HOST_MEMORY:             return "Out of host memory";
  case CL_PROFILING_INFO_NOT_AVAILABLE:   return "Profiling information not available";
  case CL_MEM_COPY_OVERLAP:               return "Memory copy overlap";
  case CL_IMAGE_FORMAT_MISMATCH:          return "Image format mismatch";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:     return "Image format not supported";
  case CL_BUILD_PROGRAM_FAILURE:          return "Program build failure";
  case CL_MAP_FAILURE:                    return "Map failure";
#ifdef CL_VERSION_1_1
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:   return "Buffer offset improperly aligned";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "Event in wait list has an error status";
#endif
  case CL_INVALID_VALUE:                  return "Invalid value";
  case CL_INVALID_DEVICE_TYPE:            return "Invalid device type";
  case CL_INVALID_PLATFORM:               return "Invalid platform";
  case CL_INVALID_DEVICE:                 return "Invalid device";
  case CL_INVALID_CONTEXT:                return "Invalid context";
  case CL_INVALID_QUEUE_PROPERTIES:       return "Invalid queue properties";
  case CL_INVALID_COMMAND_QUEUE:          return "Invalid command queue";
  case CL_INVALID_HOST_PTR:               return "Invalid host pointer";
  case CL_INVALID_MEM_OBJECT:             return "Invalid memory object";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:return "Invalid image format descriptor";
  case CL_INVALID_IMAGE_SIZE:             return "Invalid image size";
  case CL_INVALID_SAMPLER:                return "Invalid sampler";
  case CL_INVALID_BINARY:                 return "Invalid binary";
  case CL_INVALID_BUILD_OPTIONS:          return "Invalid build options";
  case CL_INVALID_PROGRAM:                return "Invalid program";
  case CL_INVALID_PROGRAM_EXECUTABLE:     return "Invalid program executable";
  case CL_INVALID_KERNEL_NAME:            return "Invalid kernel name";
  case CL_INVALID_KERNEL_DEFINITION:      return "Invalid kernel definition";
  case CL_INVALID_KERNEL:                 return "Invalid kernel";
  case CL_INVALID_ARG_INDEX:              return "Invalid argument index";
  case CL_INVALID_ARG_VALUE:              return "Invalid argument value";
  case CL_INVALID_ARG_SIZE:               return "Invalid argument size";
  case CL_INVALID_KERNEL_ARGS:            return "Invalid kernel arguments";
  case CL_INVALID_WORK_DIMENSION:         return "Invalid work dimension";
  case CL_INVALID_WORK_GROUP_SIZE:        return "Invalid work group size";
  case CL_INVALID_WORK_ITEM_SIZE:         return "Invalid work item size";
  case CL_INVALID_GLOBAL_OFFSET:          return "Invalid global offset";
  case CL_INVALID_EVENT_WAIT_LIST:        return "Invalid event wait list";
  case CL_INVALID_EVENT:                  return "Invalid event";
  case CL_INVALID_OPERATION:              return "Invalid operation";
  case CL_INVALID_GL_OBJECT:              return "Invalid OpenGL object";
  case CL_INVALID_BUFFER_SIZE:            return "Invalid buffer size";
  case CL_INVALID_MIP_LEVEL:              return "Invalid mip-map level";
  case CL_INVALID_GLOBAL_WORK_SIZE:       return "Invalid global work size";
#ifdef CL_VERSION_1_1
  case CL_INVALID_PROPERTY:               return "Invalid property";
#endif
  default: return "Unknown error";
  }
}

static int check_ext(cl_ctx *ctx, const char *name) {
  cl_device_id dev;
  size_t sz;
  int res = 0;

  if (ctx->exts == NULL) {
    dev = get_dev(ctx->ctx, &res);
    if (dev == NULL) return res;

    ctx->err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, NULL, &sz);
    if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;

    ctx->exts = malloc(sz);
    if (ctx->exts == NULL) return GA_MEMORY_ERROR;

    ctx->err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sz, ctx->exts, NULL);
    if (ctx->err != CL_SUCCESS) {
      free(ctx->exts);
      ctx->exts = NULL;
      return GA_IMPL_ERROR;
    }
  }
  return (strstr(ctx->exts, name) == NULL) ? GA_DEVSUP_ERROR : 0;
}

static void
#ifdef _MSC_VER
__stdcall
#endif
errcb(const char *errinfo, const void *pi, size_t cb, void *u) {
  fprintf(stderr, "%s\n", errinfo);
}

static void *cl_init(int devno, int flags, int *ret) {
  int platno;
  cl_device_id *ds;
  cl_device_id d;
  cl_platform_id *ps;
  cl_platform_id p;
  cl_uint nump, numd;
  cl_context_properties props[3] = {
    CL_CONTEXT_PLATFORM, 0,
    0,
  };
  cl_context ctx;
  cl_ctx *res;

  platno = devno >> 16;
  devno &= 0xFFFF;

  err = clGetPlatformIDs(0, NULL, &nump);
  CHKFAIL(NULL);

  if ((unsigned int)platno >= nump || platno < 0) FAIL(NULL, GA_VALUE_ERROR);

  ps = calloc(sizeof(*ps), nump);
  if (ps == NULL) FAIL(NULL, GA_MEMORY_ERROR);
  err = clGetPlatformIDs(nump, ps, NULL);
  /* We may get garbage on failure here but it won't matter as we will
     not use it */
  p = ps[platno];
  free(ps);
  CHKFAIL(NULL);

  err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &numd);
  CHKFAIL(NULL);

  if ((unsigned int)devno >= numd || devno < 0) FAIL(NULL, GA_VALUE_ERROR);

  ds = calloc(sizeof(*ds), numd);
  if (ds == NULL) FAIL(NULL, GA_MEMORY_ERROR);
  err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, numd, ds, NULL);
  d = ds[devno];
  free(ds);
  CHKFAIL(NULL);

  props[1] = (cl_context_properties)p;
  ctx = clCreateContext(props, 1, &d, errcb, NULL, &err);
  CHKFAIL(NULL);

  res = cl_make_ctx(ctx);
  clReleaseContext(ctx);
  if (res == NULL) FAIL(NULL, GA_IMPL_ERROR);  // can also be a sys_error
  return res;
}

static void cl_deinit(void *c) {
  ASSERT_CTX((cl_ctx *)c);
  cl_free_ctx((cl_ctx *)c);
}

static gpudata *cl_alloc(void *c, size_t size, void *data, int flags,
                         int *ret) {
  cl_ctx *ctx = (cl_ctx *)c;
  gpudata *res;
  void *hostp = NULL;
  cl_mem_flags clflags = CL_MEM_READ_WRITE;

  ASSERT_CTX(ctx);

  if (flags & GA_BUFFER_INIT) {
    if (data == NULL) FAIL(NULL, GA_VALUE_ERROR);
    hostp = data;
    clflags |= CL_MEM_COPY_HOST_PTR;
  }

  if (flags & GA_BUFFER_HOST) {
    clflags |= CL_MEM_ALLOC_HOST_PTR;
  }

  if (flags & GA_BUFFER_READ_ONLY) {
    if (flags & GA_BUFFER_WRITE_ONLY) FAIL(NULL, GA_VALUE_ERROR);
    clflags |= CL_MEM_READ_ONLY;
  }

  if (flags & GA_BUFFER_WRITE_ONLY) {
    if (flags & GA_BUFFER_READ_ONLY) FAIL(NULL, GA_VALUE_ERROR);
    clflags |= CL_MEM_WRITE_ONLY;
  }

  res = malloc(sizeof(*res));
  if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
  res->refcnt = 1;

  if (size == 0) {
    /* OpenCL doesn't like a zero-sized buffer */
    size = 1;
  }

  res->buf = clCreateBuffer(ctx->ctx, clflags, size, hostp, &ctx->err);
  res->ev = NULL;
  if (ctx->err != CL_SUCCESS) {
    free(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }

  res->ctx = ctx;
  ctx->refcnt++;

  TAG_BUF(res);
  return res;
}

static void cl_retain(gpudata *b) {
  ASSERT_BUF(b);
  b->refcnt++;
}

static void cl_release(gpudata *b) {
  ASSERT_BUF(b);
  b->refcnt--;
  if (b->refcnt == 0) {
    CLEAR(b);
    clReleaseMemObject(b->buf);
    if (b->ev != NULL)
      clReleaseEvent(b->ev);
    cl_free_ctx(b->ctx);
    free(b);
  }
}

static int cl_share(gpudata *a, gpudata *b, int *ret) {
#ifdef CL_VERSION_1_1
  cl_ctx *ctx;
  cl_mem aa, bb;
#endif
  ASSERT_BUF(a);
  ASSERT_BUF(b);
  if (a->buf == b->buf) return 1;
#ifdef CL_VERSION_1_1
  if (a->ctx != b->ctx) return 0;
  ctx = a->ctx;
  ASSERT_CTX(ctx);
  ctx->err = clGetMemObjectInfo(a->buf, CL_MEM_ASSOCIATED_MEMOBJECT,
				sizeof(aa), &aa, NULL);
  CHKFAIL(-1);
  ctx->err = clGetMemObjectInfo(b->buf, CL_MEM_ASSOCIATED_MEMOBJECT,
				sizeof(bb), &bb, NULL);
  CHKFAIL(-1);
  if (aa == NULL) aa = a->buf;
  if (bb == NULL) bb = b->buf;
  if (aa == bb) return 1;
#endif
  return 0;
}

static int cl_move(gpudata *dst, size_t dstoff, gpudata *src, size_t srcoff,
                   size_t sz) {
  cl_ctx *ctx;
  cl_event ev;
  cl_event evw[2];
  cl_event *evl = NULL;
  cl_uint num_ev = 0;

  ASSERT_BUF(dst);
  ASSERT_BUF(src);

  if (dst->ctx != src->ctx) return GA_VALUE_ERROR;
  ctx = dst->ctx;

  ASSERT_CTX(ctx);

  if (sz == 0) return GA_NO_ERROR;

  if (src->ev != NULL)
    evw[num_ev++] = src->ev;
  if (dst->ev != NULL)
    evw[num_ev++] = dst->ev;

  if (num_ev > 0)
    evl = evw;

  ctx->err = clEnqueueCopyBuffer(ctx->q, src->buf, dst->buf, srcoff, dstoff,
				 sz, num_ev, evl, &ev);
  if (ctx->err != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }
  if (src->ev != NULL)
    clReleaseEvent(src->ev);
  if (dst->ev != NULL)
    clReleaseEvent(dst->ev);

  src->ev = ev;
  dst->ev = ev;
  clRetainEvent(ev);

  return GA_NO_ERROR;
}

static int cl_read(void *dst, gpudata *src, size_t srcoff, size_t sz) {
  cl_ctx *ctx = src->ctx;
  cl_event ev[1];
  cl_event *evl = NULL;
  cl_uint num_ev = 0;

  ASSERT_BUF(src);
  ASSERT_CTX(ctx);

  if (sz == 0) return GA_NO_ERROR;

  if (src->ev != NULL) {
    ev[0] = src->ev;
    evl = ev;
    num_ev = 1;
  }

  ctx->err = clEnqueueReadBuffer(ctx->q, src->buf, CL_TRUE, srcoff, sz, dst,
				 num_ev, evl, NULL);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;
  if (src->ev != NULL) clReleaseEvent(src->ev);
  src->ev = NULL;

  return GA_NO_ERROR;
}

static int cl_write(gpudata *dst, size_t dstoff, const void *src, size_t sz) {
  cl_ctx *ctx = dst->ctx;
  cl_event ev[1];
  cl_event *evl = NULL;
  cl_uint num_ev = 0;

  ASSERT_BUF(dst);
  ASSERT_CTX(ctx);

  if (sz == 0) return GA_NO_ERROR;

  if (dst->ev != NULL) {
    ev[0] = dst->ev;
    evl = ev;
    num_ev = 1;
  }

  ctx->err = clEnqueueWriteBuffer(ctx->q, dst->buf, CL_TRUE, dstoff, sz, src,
				  num_ev, evl, NULL);
  if (err != CL_SUCCESS) return GA_IMPL_ERROR;
  if (dst->ev != NULL) clReleaseEvent(dst->ev);
  dst->ev = NULL;

  return GA_NO_ERROR;
}

static int cl_memset(gpudata *dst, size_t offset, int data) {
  char local_kern[256];
  cl_ctx *ctx = dst->ctx;
  const char *rlk[1];
  void *args[1];
  size_t sz, bytes, n, ls[2], gs[2];
  gpukernel *m;
  cl_mem_flags fl;
  int type;
  int r, res = GA_IMPL_ERROR;

  unsigned char val = (unsigned char)data;
  cl_uint pattern = (cl_uint)val & (cl_uint)val >> 8 & \
    (cl_uint)val >> 16 & (cl_uint)val >> 24;

  ASSERT_BUF(dst);
  ASSERT_CTX(ctx);

  ctx->err = clGetMemObjectInfo(dst->buf, CL_MEM_FLAGS, sizeof(fl), &fl, NULL);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;

  if (fl & CL_MEM_READ_ONLY) return GA_READONLY_ERROR;

  ctx->err = clGetMemObjectInfo(dst->buf, CL_MEM_SIZE, sizeof(bytes), &bytes,
				NULL);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;

  bytes -= offset;

  if (bytes == 0) return GA_NO_ERROR;

  if ((bytes % 16) == 0) {
    n = bytes/16;
    r = snprintf(local_kern, sizeof(local_kern),
                 "__kernel void kmemset(__global uint4 *mem) {"
                 "unsigned int i; __global char *tmp = (__global char *)mem;"
                 "tmp += %" SPREFIX "u; mem = (__global uint4 *)tmp;"
                 "for (i = get_global_id(0); i < %" SPREFIX "u; "
                 "i += get_global_size(0)) {mem[i] = (uint4)(%u,%u,%u,%u); }}",
                 offset, n, pattern, pattern, pattern, pattern);
  } else if ((bytes % 8) == 0) {
    n = bytes/8;
    r = snprintf(local_kern, sizeof(local_kern),
                 "__kernel void kmemset(__global uint2 *mem) {"
                 "unsigned int i; __global char *tmp = (__global char *)mem;"
                 "tmp += %" SPREFIX "u; mem = (__global uint2 *)tmp;"
                 "for (i = get_global_id(0); i < %" SPREFIX "u;"
                 "i += get_global_size(0)) {mem[i] = (uint2)(%u,%u); }}",
                 offset, n, pattern, pattern);
  } else if ((bytes % 4) == 0) {
    n = bytes/4;
    r = snprintf(local_kern, sizeof(local_kern),
                 "__kernel void kmemset(__global unsigned int *mem) {"
                 "unsigned int i; __global char *tmp = (__global char *)mem;"
                 "tmp += %" SPREFIX "u; mem = (__global unsigned int *)tmp;"
                 "for (i = get_global_id(0); i < %" SPREFIX "u;"
                 "i += get_global_size(0)) {mem[i] = %u; }}",
                 offset, n, pattern);
  } else {
    if (check_ext(ctx, CL_SMALL))
      return GA_DEVSUP_ERROR;
    n = bytes;
    r = snprintf(local_kern, sizeof(local_kern),
                 "__kernel void kmemset(__global unsigned char *mem) {"
                 "unsigned int i; mem += %" SPREFIX "u;"
                 "for (i = get_global_id(0); i < %" SPREFIX "u;"
                 "i += get_global_size(0)) {mem[i] = %u; }}",
                 offset, n, val);
  }
  /* If this assert fires, increase the size of local_kern above. */
  assert(r <= sizeof(local_kern));

  sz = strlen(local_kern);
  rlk[0] = local_kern;
  type = GA_BUFFER;

  m = cl_newkernel(ctx, 1, rlk, &sz, "kmemset", 1, &type, 0, &res, NULL);
  if (m == NULL) return res;

  /* Cheap kernel scheduling */
  res = cl_property(NULL, NULL, m, GA_KERNEL_PROP_MAXLSIZE, &ls[0]);
  if (res != GA_NO_ERROR) goto fail;
  gs[0] = ((n-1) / ls[0]) + 1;
  gs[1] = ls[1] = 1;
  args[0] = dst;
  res = cl_callkernel(m, ls, gs, args);

 fail:
  cl_releasekernel(m);
  return res;
}

static int cl_check_extensions(const char **preamble, unsigned int *count,
                               int flags, cl_ctx *ctx) {
  if (flags & GA_USE_CLUDA) {
    preamble[*count] = CL_PREAMBLE;
    (*count)++;
  }
  if (flags & GA_USE_SMALL) {
    if (check_ext(ctx, CL_SMALL)) return GA_DEVSUP_ERROR;
    preamble[*count] = PRAGMA CL_SMALL ENABLE;
    (*count)++;
  }
  if (flags & GA_USE_DOUBLE) {
    if (check_ext(ctx, CL_DOUBLE)) return GA_DEVSUP_ERROR;
    preamble[*count] = PRAGMA CL_DOUBLE ENABLE;
    (*count)++;
  }
  if (flags & GA_USE_COMPLEX) {
    return GA_DEVSUP_ERROR; // for now
  }
  if (flags & GA_USE_HALF) {
    if (check_ext(ctx, CL_HALF)) return GA_DEVSUP_ERROR;
    preamble[*count] = PRAGMA CL_HALF ENABLE;
    (*count)++;
  }
  if (flags & (GA_USE_PTX|GA_USE_CUDA)) {
    return GA_DEVSUP_ERROR;
  }
  return GA_NO_ERROR;
}

static void gpukernel_source_with_line_numbers(unsigned int count, const char **news, size_t *newl, 
                                              strb *src) {
  assert(src != NULL);
  unsigned int section, line, i, j, len;

  line=1;  // start the line counter at 1
  for(section=0; section<count; section++) {
    len = (int)newl[section];
    if(len<=0)
      len=strlen(news[section]);

    i=0; // position of line-starts within news[section]
    while(i<len) {
      strb_appendf(src, "%04d\t", line);

      for(j=i; j<len && news[section][j] != '\n'; j++); // look for next line-end
      strb_appendn(src, news[section]+i, (j-i));
      strb_appendc(src, '\n');

      i = j+1;  // Character after the newline
      line++;
    }
  }
}

static gpukernel *cl_newkernel(void *c, unsigned int count,
                               const char **strings, const size_t *lengths,
                               const char *fname, unsigned int argcount,
                               const int *types, int flags, int *ret, char **err_str) {
  cl_ctx *ctx = (cl_ctx *)c;
  gpukernel *res;
  cl_device_id dev;
  cl_program p;
  // Sync this table size with the number of flags that can add stuff
  // at the beginning
  const char *preamble[4];
  size_t *newl = NULL;
  const char **news = NULL;
  unsigned int n = 0;
  int error;

  ASSERT_CTX(ctx);

  if (count == 0) FAIL(NULL, GA_VALUE_ERROR);

  dev = get_dev(ctx->ctx, ret);
  if (dev == NULL) return NULL;

  if (flags & GA_USE_BINARY) {
    // GA_USE_BINARY is exclusive
    if (flags & ~GA_USE_BINARY)
      FAIL(NULL, GA_INVALID_ERROR);
    // We need the length for binary data and there is only one blob.
    if (count != 1 || lengths == NULL || lengths[0] == 0)
      FAIL(NULL, GA_VALUE_ERROR);
    p = clCreateProgramWithBinary(ctx->ctx, 1, &dev, lengths, (const unsigned char **)strings, NULL, &ctx->err);
    if (ctx->err != CL_SUCCESS) {
      clReleaseProgram(p);
      FAIL(NULL, GA_IMPL_ERROR);
    }
  } else {

    error = cl_check_extensions(preamble, &n, flags, ctx);
    if (error != GA_NO_ERROR) FAIL(NULL, error);

    if (n != 0) {
      news = calloc(count+n, sizeof(const char *));
      if (news == NULL) {
        FAIL(NULL, GA_SYS_ERROR);
      }
      memcpy(news, preamble, n*sizeof(const char *));
      memcpy(news+n, strings, count*sizeof(const char *));
      if (lengths == NULL) {
        newl = NULL;
      } else {
        newl = calloc(count+n, sizeof(size_t));
        if (newl == NULL) {
          free(news);
          FAIL(NULL, GA_MEMORY_ERROR);
        }
        memcpy(newl+n, lengths, count*sizeof(size_t));
      }
    } else {
      news = strings;
      newl = (size_t *)lengths;
    }

    p = clCreateProgramWithSource(ctx->ctx, count+n, news, newl, &ctx->err);
    if (ctx->err != CL_SUCCESS) {
      if (n != 0) {
        free(news);
        free(newl);
      }
      FAIL(NULL, GA_IMPL_ERROR);
    }
  }

  ctx->err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
  if (ctx->err != CL_SUCCESS) {
    if (ctx->err == CL_BUILD_PROGRAM_FAILURE && err_str!=NULL) {
      *err_str = NULL;  // Fallback, in case there's an error

      strb debug_msg = STRB_STATIC_INIT;
      // We're substituting debug_msg for a string with this first line:
      strb_appends(&debug_msg, "Program build failure ::\n"); 

      // Determine the size of the log
      size_t log_size;
      clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

      strb_ensure(&debug_msg, debug_msg.l + log_size);
      if(!strb_error(&debug_msg) && log_size>=1) { // Make sure the strb is in a valid state
        // Get the log directly into the debug_msg
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size, debug_msg.s+debug_msg.l, NULL);
        debug_msg.l += (log_size-1); 
      }

      if (flags & GA_USE_BINARY) {
        // Not clear what to do with binary 'source' - the log will have to suffice
      } else {
        gpukernel_source_with_line_numbers(count+n, news, newl, &debug_msg);
      }

      strb_append0(&debug_msg); // Make sure a final '\0' is present
      
      if(!strb_error(&debug_msg)) { // Make sure the strb is in a valid state
        *err_str = strndup(debug_msg.s, debug_msg.l);
        if(*err_str == NULL) {
          clReleaseProgram(p);
          if (n != 0) {
            free(news);
            free(newl);
          }
          FAIL(NULL, GA_MEMORY_ERROR);
        }
      }

      //free(*err_str);  // *err_str must be free()d by the caller (see docs in kernel.h)
    }

    clReleaseProgram(p);
    if (n != 0) {
      free(news);
      free(newl);
    }
    FAIL(NULL, GA_IMPL_ERROR);
  }

  if (n != 0) {
    free(news);
    free(newl);
  }

  res = malloc(sizeof(*res));
  if (res == NULL) FAIL(NULL, GA_MEMORY_ERROR);
  res->refcnt = 1;
  res->ev = NULL;
  res->argcount = argcount;
  res->k = clCreateKernel(p, fname, &ctx->err);
  res->types = NULL;  /* This avoids a crash in cl_releasekernel */
  res->ctx = ctx;
  ctx->refcnt++;
  clReleaseProgram(p);
  TAG_KER(res);
  if (ctx->err != CL_SUCCESS) {
    cl_releasekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  res->types = calloc(argcount, sizeof(int));
  if (res->types == NULL) {
    cl_releasekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  memcpy(res->types, types, argcount * sizeof(int));

  return res;
}

static void cl_retainkernel(gpukernel *k) {
  ASSERT_KER(k);
  k->refcnt++;
}

static void cl_releasekernel(gpukernel *k) {
  ASSERT_KER(k);

  k->refcnt--;
  if (k->refcnt == 0) {
    CLEAR(k);
    if (k->ev != NULL) clReleaseEvent(k->ev);
    if (k->k) clReleaseKernel(k->k);
    cl_free_ctx(k->ctx);
    free(k->types);
    free(k);
  }
}

static int cl_callkernel(gpukernel *k, size_t ls[2], size_t gs[2],
                         void **args) {
  cl_ctx *ctx = k->ctx;
  size_t _gs[2];
  cl_event ev;
  cl_event *evw;
  gpudata *btmp;
  cl_device_id dev;
  cl_ulong temp;
  cl_uint num_ev;
  cl_uint i;
  int res = 0;

  ASSERT_KER(k);
  ASSERT_CTX(ctx);

  dev = get_dev(ctx->ctx, &res);
  if (dev == NULL) return res;

  num_ev = 0;
  evw = calloc(sizeof(cl_event), k->argcount);
  if (evw == NULL) {
    return GA_MEMORY_ERROR;
  }
  
  for (i = 0; i < k->argcount; i++) {
    if (k->types[i] == GA_BUFFER) {
      btmp = (gpudata *)args[i];
      if (btmp->ev != NULL)
        evw[num_ev++] = btmp->ev;
      ctx->err = clSetKernelArg(k->k, i, sizeof(cl_mem), &btmp->buf);
    } else if (k->types[i] == GA_SIZE) {
      temp = *((size_t *)args[i]);
      ctx->err = clSetKernelArg(k->k, i, gpuarray_get_elsize(k->types[i]),
                                &temp);
    } else {
      ctx->err = clSetKernelArg(k->k, i, gpuarray_get_elsize(k->types[i]),
                                args[i]);
    }
    if (ctx->err != CL_SUCCESS) {
      free(evw);
      return GA_IMPL_ERROR;
    }
  }

  if (num_ev == 0) {
    free(evw);
    evw = NULL;
  }

  _gs[0] = gs[0] * ls[0];
  _gs[1] = gs[1] * ls[1];
  ctx->err = clEnqueueNDRangeKernel(ctx->q, k->k, 2, NULL, _gs, ls,
				    num_ev, evw, &ev);
  free(evw);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;

  for (i = 0; i < k->argcount; i++) {
    if (k->types[i] == GA_BUFFER) {
      if (((gpudata *)args[i])->ev != NULL)
        clReleaseEvent(((gpudata *)args[i])->ev);
      ((gpudata *)args[i])->ev = ev;
      clRetainEvent(ev);
    }
  }
  if (k->ev != NULL)
    clReleaseEvent(k->ev);
  k->ev = ev;

  return GA_NO_ERROR;
}

static int cl_kernelbin(gpukernel *k, size_t *sz, void **obj) {
  cl_ctx *ctx = k->ctx;
  cl_program p;
  size_t rsz;
  void *res;

  ASSERT_KER(k);
  ASSERT_CTX(ctx);

  ctx->err = clGetKernelInfo(k->k, CL_KERNEL_PROGRAM, sizeof(p), &p, NULL);
  if (ctx->err != CL_SUCCESS)
    return GA_IMPL_ERROR;
  ctx->err = clGetProgramInfo(p, CL_PROGRAM_BINARY_SIZES, sizeof(rsz), &rsz, NULL);
  if (ctx->err != CL_SUCCESS)
    return GA_IMPL_ERROR;
  res = malloc(rsz);
  if (res == NULL)
    return GA_MEMORY_ERROR;
  ctx->err = clGetProgramInfo(p, CL_PROGRAM_BINARIES, sizeof(res), &res, NULL);
  if (ctx->err != CL_SUCCESS) {
    free(res);
    return GA_IMPL_ERROR;
  }
  *sz = rsz;
  *obj = res;
  return GA_NO_ERROR;
}

static int cl_sync(gpudata *b) {
  cl_ctx *ctx = (cl_ctx *)b->ctx;

  ASSERT_BUF(b);
  ASSERT_CTX(ctx);

  if (b->ev != NULL) {
    ctx->err = clWaitForEvents(1, &b->ev);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    clReleaseEvent(b->ev);
    b->ev = NULL;
  }
  return GA_NO_ERROR;
}

static gpudata *cl_transfer(gpudata *buf, size_t offset, size_t sz,
                            void *dst_ctx, int may_share) {
  cl_ctx *ctx = buf->ctx;

  ASSERT_BUF(buf);
  ASSERT_CTX(ctx);
  ASSERT_CTX((cl_ctx *)dst_ctx);

  if (ctx == dst_ctx && may_share && offset == 0) {
    cl_retain(buf);
    return buf;
  }
  return NULL;
}

static const char ELEM_HEADER[] = "#define DTYPEA %s\n"
  "#define DTYPEB %s\n"
  "__kernel void elemk(__global const DTYPEA *a_data,"
  "                    __global DTYPEB *b_data){"
  "const int idx = get_global_id(0);"
  "const int numThreads = get_global_size(0);"
  "__global char *tmp; tmp = (__global char *)a_data; tmp += %" SPREFIX "u;"
  "a_data = (__global const DTYPEA *)tmp; tmp = (__global char *)b_data;"
  "tmp += %" SPREFIX "u; b_data = (__global DTYPEB *)tmp;"
  "for (int i = idx; i < %" SPREFIX "u; i+= numThreads) {"
  "__global const char *a_p = (__global const char *)a_data;"
  "__global char *b_p = (__global char *)b_data;";

static const char ELEM_FOOTER[] =
  "__global const DTYPEA *a = (__global const DTYPEA *)a_p;"
  "__global DTYPEB *b = (__global DTYPEB *)b_p;"
  "b[0] = a[0];}}\n";

static int cl_extcopy(gpudata *input, size_t ioff, gpudata *output,
                      size_t ooff, int intype, int outtype, unsigned int a_nd,
                      const size_t *a_dims, const ssize_t *a_str,
                      unsigned int b_nd, const size_t *b_dims,
                      const ssize_t *b_str) {
  cl_ctx *ctx = input->ctx;
  strb sb = STRB_STATIC_INIT;
  size_t nEls, ls[2], gs[2];
  gpukernel *k;
  void *args[2];
  cl_mem_flags fl;
  int res = GA_SYS_ERROR;
  unsigned int i;
  int flags = GA_USE_CLUDA;
  int types[2];

  ASSERT_BUF(input);
  ASSERT_BUF(output);
  ASSERT_CTX(ctx);

  if (input->ctx != output->ctx) return GA_VALUE_ERROR;

  ctx->err = clGetMemObjectInfo(input->buf, CL_MEM_FLAGS, sizeof(fl), &fl,
                                NULL);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;
  if (fl & CL_MEM_WRITE_ONLY) return GA_WRITEONLY_ERROR;

  ctx->err = clGetMemObjectInfo(output->buf, CL_MEM_FLAGS, sizeof(fl), &fl,
                                NULL);
  if (ctx->err != CL_SUCCESS) return GA_IMPL_ERROR;
  if (fl & CL_MEM_READ_ONLY) return GA_READONLY_ERROR;

  nEls = 1;
  for (i = 0; i < a_nd; i++) {
    nEls *= a_dims[i];
  }

  if (nEls == 0) return GA_NO_ERROR;

  if (outtype == GA_DOUBLE || intype == GA_DOUBLE ||
      outtype == GA_CDOUBLE || intype == GA_CDOUBLE) {
    flags |= GA_USE_DOUBLE;
  }

  if (outtype == GA_HALF || intype == GA_HALF) {
    flags |= GA_USE_HALF;
  }

  if (gpuarray_get_elsize(outtype) < 4 || gpuarray_get_elsize(intype) < 4) {
    /* Should check for non-mod4 strides too */
    flags |= GA_USE_SMALL;
  }

  if (outtype == GA_CFLOAT || intype == GA_CFLOAT ||
      outtype == GA_CDOUBLE || intype == GA_CDOUBLE) {
    flags |= GA_USE_COMPLEX;
  }

  strb_appendf(&sb, ELEM_HEADER,
	       gpuarray_get_type(intype)->cluda_name,
	       gpuarray_get_type(outtype)->cluda_name,
	       ioff, ooff, nEls);

  gpuarray_elem_perdim(&sb, a_nd, a_dims, a_str, "a_p");
  gpuarray_elem_perdim(&sb, b_nd, b_dims, b_str, "b_p");

  strb_appends(&sb, ELEM_FOOTER);

  if (strb_error(&sb))
    goto fail;

  types[0] = types[1] = GA_BUFFER;
  k = cl_newkernel(ctx, 1, (const char **)&sb.s, &sb.l, "elemk",
                   2, types, flags, &res, NULL);
  if (k == NULL) goto fail;
  /* Cheap kernel scheduling */
  res = cl_property(NULL, NULL, k, GA_KERNEL_PROP_MAXLSIZE, &ls[0]);
  if (res != GA_NO_ERROR) goto kfail;

  gs[0] = ((nEls-1) / ls[0]) + 1;
  gs[1] = ls[1] = 1;
  args[0] = input;
  args[1] = output;
  res = cl_callkernel(k, ls, gs, args);

 kfail:
  cl_releasekernel(k);
 fail:
  strb_clear(&sb);
  return res;
}

#ifdef WITH_OPENCL_CLBLAS
extern gpuarray_blas_ops clblas_ops;
#endif

static int cl_property(void *c, gpudata *buf, gpukernel *k, int prop_id,
                       void *res) {
  cl_ctx *ctx = NULL;
  if (c != NULL) {
    ctx = (cl_ctx *)c;
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
    size_t sz;
    size_t *psz;
    cl_device_id id;
    cl_uint ui;

  case GA_CTX_PROP_DEVNAME:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &sz);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    s = malloc(sz);
    if (s == NULL)
      return GA_MEMORY_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_NAME, sz, s, NULL);
    if (ctx->err != CL_SUCCESS) {
      free(s);
      return GA_IMPL_ERROR;
    }
    *((char **)res) = s;
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXLSIZE:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL,
                               &sz);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    psz = malloc(sz);
    if (psz == NULL)
      return GA_MEMORY_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sz, psz, NULL);
    if (ctx->err != CL_SUCCESS) {
      free(psz);
      return GA_IMPL_ERROR;
    }
    *((size_t *)res) = psz[0];
    free(psz);
    return GA_NO_ERROR;

  case GA_CTX_PROP_LMEMSIZE:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(sz), &sz,
                               NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    *((size_t *)res) = sz;
    return GA_NO_ERROR;

  case GA_CTX_PROP_NUMPROCS:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(ui),
                               &ui, NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    *((unsigned int *)res) = ui;
    return GA_NO_ERROR;

  case GA_CTX_PROP_MAXGSIZE:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id), &id,
                                NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_ADDRESS_BITS, sizeof(ui), &ui,
                               NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    ctx->err = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(sz),
                               &sz, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    if (ui == 32) {
      sz = 4294967295UL/sz;
    } else if (ui == 64) {
      sz = 18446744073709551615ULL/sz;
    } else {
      assert(0 && "This should not be reached!");
    }
    *((size_t *)res) = sz;
    return GA_NO_ERROR;

  case GA_CTX_PROP_BLAS_OPS:
#ifdef WITH_OPENCL_CLBLAS
    *((gpuarray_blas_ops **)res) = &clblas_ops;
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
    ctx->err = clGetMemObjectInfo(buf->buf, CL_MEM_SIZE, sizeof(sz), &sz,
                                  NULL);
    if (ctx->err != CL_SUCCESS)
      return GA_IMPL_ERROR;
    *((size_t *)res) = sz;
    return GA_NO_ERROR;

  /* GA_BUFFER_PROP_CTX is not ordered to simplify code */
  case GA_BUFFER_PROP_CTX:
  case GA_KERNEL_PROP_CTX:
    *((void **)res) = (void *)ctx;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_MAXLSIZE:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    ctx->err = clGetKernelWorkGroupInfo(k->k, id, CL_KERNEL_WORK_GROUP_SIZE,
                                        sizeof(sz), &sz, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    *((size_t *)res) = sz;
    return GA_NO_ERROR;

  case GA_KERNEL_PROP_PREFLSIZE:
    ctx->err = clGetContextInfo(ctx->ctx, CL_CONTEXT_DEVICES, sizeof(id),
                                &id, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
#ifdef OPENCL_1_1
    ctx->err = clGetKernelWorkGroupInfo(k->k, id,
                                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                        sizeof(sz), &sz, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
#else
    ctx->err = clGetKernelWorkGroupInfo(k->k, id, CL_KERNEL_WORK_GROUP_SIZE,
                                        sizeof(sz), &sz, NULL);
    if (ctx->err != GA_NO_ERROR)
      return GA_IMPL_ERROR;
    /*
      This is sort of a guess, AMD generally has 64 and NVIDIA has 32.
      Since this is a multiple, it would not hurt a lot to overestimate
      unless we go over the maximum. However underestimating may hurt
      performance due to the way we do the automatic allocation.

      Also OpenCL 1.0 kind of sucks and this is only used for that.
    */
    sz = (64 < sz) ? 64 : sz;
#endif
    *((size_t *)res) = sz;
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

static const char *cl_error(void *c) {
  cl_ctx *ctx = (cl_ctx *)c;
  if (ctx == NULL)
    return get_error_string(err);
  else
    ASSERT_CTX(ctx);
    return get_error_string(ctx->err);
}

GPUARRAY_LOCAL
const gpuarray_buffer_ops opencl_ops = {cl_init,
                                       cl_deinit,
                                       cl_alloc,
                                       cl_retain,
                                       cl_release,
                                       cl_share,
                                       cl_move,
                                       cl_read,
                                       cl_write,
                                       cl_memset,
                                       cl_newkernel,
                                       cl_retainkernel,
                                       cl_releasekernel,
                                       cl_callkernel,
                                       cl_kernelbin,
                                       cl_sync,
                                       cl_extcopy,
                                       cl_transfer,
                                       cl_property,
                                       cl_error};
