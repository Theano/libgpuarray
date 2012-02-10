#include "compyte_buffer.h"

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else

#include <CL/opencl.h>

#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* To work around the lack of byte addressing */
#define MIN_SIZE_INCR 4

struct _gpudata {
  cl_mem buf;
  cl_command_queue q;
  /* Use subbuffers in OpenCL 1.1 to work around the need for an offset */
  size_t offset;
};

struct _gpukernel {
  cl_program p;
  cl_kernel k;
  cl_command_queue q;
};

#define FAIL(v, e) { if (ret) *ret = e; return v; }
#define CHKFAIL(v) if (err != CL_SUCCESS) FAIL(v, GA_IMPL_ERROR)

static cl_int err;

static gpukernel *cl_newkernel(void *ctx, unsigned int count,
			       const char **strings, const size_t *lengths,
			       const char *fname, int *ret);
static void cl_freekernel(gpukernel *k);
static int cl_setkernelarg(gpukernel *k, unsigned int index,
			   size_t sz, const void *val);
static int cl_setkernelargbuf(gpukernel *k, unsigned int index,
			      gpudata *b);
static int cl_callkernel(gpukernel *k, unsigned int, unsigned int,
			 unsigned int, unsigned int, unsigned int,
			 unsigned int);

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
  default: return "Unknown error";
  }
}

static cl_command_queue make_q(cl_context ctx, int *ret) {
  size_t sz;
  cl_device_id *ids;
  cl_command_queue res;

  err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz);
  CHKFAIL(NULL);
  
  ids = malloc(sz);
  if (ids == NULL) FAIL(NULL, GA_MEMORY_ERROR);
  
  if ((err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sz, ids, NULL)) != CL_SUCCESS) {
    free(ids);
    FAIL(NULL, GA_IMPL_ERROR);
  }

  res = clCreateCommandQueue(ctx, ids[0], 0, &err);
  free(ids);
  if (err != CL_SUCCESS) {
    free(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  return res;
}

static void *cl_init(int dev, int *ret) {
  /* XXX: implement */
  return NULL;
}

static gpudata *cl_alloc(void *ctx, size_t size, int *ret)
{
  gpudata *res;

  /* OpenCL does not always support byte addressing
     so fudge size to work around that */
  if (size % 4)
    size += MIN_SIZE_INCR-(size % MIN_SIZE_INCR);

  res = malloc(sizeof(*res));
  if (res == NULL) FAIL(NULL, GA_SYS_ERROR);
  
  res->q = make_q((cl_context)ctx, ret);
  if (res->q == NULL) {
    free(res);
    return NULL;
  }

  res->buf = clCreateBuffer((cl_context)ctx, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    clReleaseCommandQueue(res->q);
    free(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  res->offset = 0;
  
  return res;
}

static void cl_free(gpudata *b) {
  clReleaseCommandQueue(b->q);
  clReleaseMemObject(b->buf);
  free(b);
}

static int cl_move(gpudata *dst, gpudata *src, size_t sz) {
  cl_event ev;

  if ((err = clEnqueueCopyBuffer(dst->q, src->buf, dst->buf,
				 src->offset, dst->offset,
				 sz, 0, NULL, &ev)) != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  if ((err = clWaitForEvents(1, &ev)) != CL_SUCCESS) {
    clReleaseEvent(ev);
    return GA_IMPL_ERROR;
  }
  clReleaseEvent(ev);

  return GA_NO_ERROR;
}

static int cl_read(void *dst, gpudata *src, size_t sz) {
  if ((err = clEnqueueReadBuffer(src->q, src->buf, CL_TRUE,
				 src->offset, sz, dst,
				 0, NULL, NULL)) != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  return GA_NO_ERROR;
}

static int cl_write(gpudata *dst, void *src, size_t sz) {
  if ((err = clEnqueueReadBuffer(dst->q, dst->buf, CL_TRUE,
				 dst->offset, sz, src,
				 0, NULL, NULL)) != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  return GA_NO_ERROR;
}

static int cl_memset(gpudata *dst, int data, size_t bytes) {
  char local_kern[92];
  const char *rlk[1];
  size_t sz;
  int r, res = GA_IMPL_ERROR;

  cl_context ctx;

  /* OpenCL devices do not always support byte-addressable storage
     and stuff will break when used in this way */
  bytes = (bytes+3) / 4;

  unsigned char val = (unsigned)data;
  unsigned int pattern = (unsigned int)val & (unsigned int)val >> 8 & (unsigned int)val >> 16 & (unsigned int)val >> 24;

  err = clGetCommandQueueInfo(dst->q, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, NULL);
  if (err != CL_SUCCESS)
    return GA_IMPL_ERROR;

  r = snprintf(local_kern, sizeof(local_kern),
	       "__kernel void memset(__global unsigned int *mem) { mem[get_global_id(0)] = %u; }", pattern);
  /* If this assert fires, increase the size of local_kern above. */
  assert(r <= sizeof(local_kern));

  sz = strlen(local_kern);
  rlk[0] = local_kern;

  gpukernel *m = cl_newkernel(ctx, 1, rlk, &sz, "memset", &res);
  if (m == NULL) return res;
  res = cl_setkernelargbuf(m, 0, dst);
  if (res != GA_NO_ERROR) goto fail;
  res = cl_callkernel(m, 0, 0, 0, 0, 0, 0);

 fail:
  cl_freekernel(m);
  return res;
}

static int cl_offset(gpudata *b, int off) {
  /* check for overflow (int and size_t) */
  if (off < 0) {
    /* negative */
    if (((off == INT_MIN) && (b->offset <= INT_MAX)) || 
	(-off > b->offset)) {
      return GA_VALUE_ERROR;
    }
  } else {
    /* positive */
    if ((SIZE_MAX - off) < b->offset) {
      return GA_VALUE_ERROR;
    }
  }
  b->offset += off;
  return GA_NO_ERROR;
}

static gpukernel *cl_newkernel(void *ctx, unsigned int count, 
			       const char **strings, const size_t *lengths,
			       const char *fname, int *ret) {
  gpukernel *res;
  cl_device_id dev;

  if (count == 0) FAIL(NULL, GA_VALUE_ERROR);

  res = malloc(sizeof(*res));
  if (res == NULL) FAIL(NULL, GA_SYS_ERROR);

  res->q = make_q((cl_context)ctx, ret);
  
  err = clGetCommandQueueInfo(res->q,CL_QUEUE_DEVICE,sizeof(dev),&dev,NULL);
  if (err != CL_SUCCESS) {
    cl_freekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  
  res->p = clCreateProgramWithSource((cl_context)ctx, count, strings,
				     lengths, &err);
  if (err != CL_SUCCESS) {
    cl_freekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }

  err = clBuildProgram(res->p, 1, &dev, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    cl_freekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }  

  res->k = clCreateKernel(res->p, fname, &err);
  if (err != CL_SUCCESS) {
    cl_freekernel(res);
    FAIL(NULL, GA_IMPL_ERROR);
  }
  
  return res;
}

static void cl_freekernel(gpukernel *k) {
  if (k->q) clReleaseCommandQueue(k->q);
  if (k->k) clReleaseKernel(k->k);
  if (k->p) clReleaseProgram(k->p);
  free(k);
}

static int cl_setkernelarg(gpukernel *k, unsigned int index, size_t sz,
			   const void *val) {
  err = clSetKernelArg(k->k, index, sz, val);
  if (err != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }
  return GA_NO_ERROR;
}

static int cl_setkernelargbuf(gpukernel *k, unsigned int index, gpudata *b) {
  return cl_setkernelarg(k, index, sizeof(cl_mem), &b->buf);
}

static int cl_callkernel(gpukernel *k, unsigned int gx, unsigned int gy,
			 unsigned int gz, unsigned int lx, unsigned int ly,
			 unsigned int lz) {
  size_t gs[3], ls[3];
  cl_event ev;
  size_t *gsp, *lsp;
  
  if ((lx == 0) && (lx == ly) && (lx == lz)) {
    lsp = NULL;
  } else {
    ls[0] = lx, gs[1] = ly, gs[2] = lz;
    lsp = ls;
  }

  err = clEnqueueNDRangeKernel(k->q, k->k, 3, NULL, gsp, lsp, 0, NULL, &ev);
  if (err != CL_SUCCESS) return GA_IMPL_ERROR;

  err = clWaitForEvents(1, &ev);
  clReleaseEvent(ev);
  if (err != CL_SUCCESS) return GA_IMPL_ERROR;

  return GA_NO_ERROR;
}

static const char ELEM_HEADER[] = "#define DTYPEA %s\n"
  "#define DTYPEB %s\n"
  "__kernel void elemk(__global const DTYPEA *a_data,"
  "                    __global DTYPEB *b_data){"
  "const int idx = get_global_id(0);"
  "const int numThreads = get_global_size(0);"
  "for (int i = idx; i < %zu; i+= numThreads) {"
  "const DTYPE *a = a_data;"
  "DTYPEB *b = b_data;";

static const char ELEM_FOOTER[] = "}}\n";

static int cl_elemwise(gpudata *input, gpudata *output, int intype,
		       int outtype, const char *op, unsigned int a_nd,
		       const size_t *a_dims, const ssize_t *a_str,
		       unsigned int b_nd, const size_t *b_dims,
		       const ssize_t *b_str) {
  char *strs[64];
  unsigned int count = 0;
  int res = GA_SYS_ERROR;
  unsigned int i;

  size_t nEls = 1;
  for (i = 0; i < a_nd; i++) {
    nEls *= a_dims[i];
  }

  if (asprintf(&strs[count], ELEM_HEADER,
	       compyte_get_type(intype)->cl_name,
	       compyte_get_type(outtype)->cl_name,
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

  gpukernel *k = cl_newkernel(NULL, count, (const char **)strs, NULL, "elemk",
			      &res);
  if (k == NULL) goto fail;
  res = cl_setkernelargbuf(k, 0, input);
  if (res != GA_NO_ERROR) goto kfail;
  res = cl_setkernelargbuf(k, 1, output);
  if (res != GA_NO_ERROR) goto kfail;

  res = cl_callkernel(k, nEls, 1, 1, 0, 0, 0);

 kfail:
  cl_freekernel(k);
 fail:
  for (i = 0; i< count; i++) {
    free(strs[i]);
  }
  return res;
}

static const char *cl_error(void) {
  return get_error_string(err);
}

compyte_buffer_ops opencl_ops = {cl_init, cl_alloc, cl_free, cl_move, cl_read, cl_write, cl_memset, cl_offset, cl_newkernel, cl_freekernel, cl_setkernelarg, cl_setkernelargbuf, cl_callkernel, cl_elemwise, cl_error};
