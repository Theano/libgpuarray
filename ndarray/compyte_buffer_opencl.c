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

static cl_int err;

struct _gpudata {
  cl_mem buf;
  cl_command_queue q;
  /* Use subbuffers in OpenCL 1.1 to work around the need for an offset */
  size_t offset;
};

static const char *get_error_string(cl_int err) {
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

static gpudata *cl_alloc(void *ctx, size_t size)
{
  gpudata *res;
  cl_int err;
  cl_device_id *ids;
  size_t sz;

  /* OpenCL do not always support byte addressing
     so fudge size to work around that */
  if (size % 4)
    size += MIN_SIZE_INCR-(size % MIN_SIZE_INCR);

  if ((err = clGetContextInfo((cl_context)ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz)) != CL_SUCCESS)
    return NULL;
  
  ids = malloc(sz);
  if (ids == NULL)
    return NULL;

  if ((err = clGetContextInfo((cl_context)ctx, CL_CONTEXT_DEVICES, sz, ids, NULL)) != CL_SUCCESS) {
    free(ids);
    return NULL;
  }

  res = malloc(sizeof(*res));
  if (res == NULL) {
    free(ids);
    return NULL;
  }

  res->q = clCreateCommandQueue((cl_context)ctx, ids[0], 0, &err);
  free(ids);
  if (err != CL_SUCCESS) {
    free(res);
    return NULL;
  }
  res->buf = clCreateBuffer((cl_context)ctx, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    clReleaseCommandQueue(res->q);
    free(res);
    return NULL;
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

  return 0;
}

static int cl_read(void *dst, gpudata *src, size_t sz) {
  if ((err = clEnqueueReadBuffer(src->q, src->buf, CL_TRUE,
				 src->offset, sz, dst,
				 0, NULL, NULL)) != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  return 0;
}

static int cl_write(gpudata *dst, void *src, size_t sz) {
  if ((err = clEnqueueReadBuffer(dst->q, dst->buf, CL_TRUE,
				 dst->offset, sz, src,
				 0, NULL, NULL)) != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  return 0;
}

static int cl_memset(gpudata *dst, int data, size_t bytes) {
  char local_kern[92];
  const char *rlk[1];
  size_t sz;
  int r, res = GA_IMPL_ERROR;

  cl_event ev;
  cl_program p;
  cl_kernel k;
  cl_context ctx;
  cl_device_id dev;

  /* OpenCL devices do not always support byte-addressable storage
     and stuff will break when used in this way */
  bytes = (bytes+3) / 4;

  unsigned char val = (unsigned)data;
  unsigned int pattern = (unsigned int)val & (unsigned int)val >> 8 & (unsigned int)val >> 16 & (unsigned int)val >> 24;

  err = clGetCommandQueueInfo(dst->q, CL_QUEUE_CONTEXT, sizeof(ctx), &ctx, NULL);
  if (err != CL_SUCCESS)
    return GA_IMPL_ERROR;

  err = clGetCommandQueueInfo(dst->q, CL_QUEUE_DEVICE, sizeof(dev), &dev, NULL);
  if (err != CL_SUCCESS)
    return GA_IMPL_ERROR;

  r = snprintf(local_kern, sizeof(local_kern),
	       "__kernel void memset(__global unsigned int *mem) { mem[get_global_id(0)] = %u; }", pattern);
  /* If this assert fires, increase the size of local_kern above. */ 
  assert(r <= sizeof(local_kern));

  sz = strlen(local_kern);
  rlk[0] = local_kern;
  p = clCreateProgramWithSource(ctx, 1, rlk, &sz, &err);
  if (err != CL_SUCCESS) {
    return GA_IMPL_ERROR;
  }

  if (clBuildProgram(p, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) {
    goto fail_prog;
  }

  k = clCreateKernel(p, "memset", &err);
  if (err != CL_SUCCESS) {
    goto fail_prog;
  }

  if (clSetKernelArg(k, 0, sizeof(cl_mem), &dst->buf) != CL_SUCCESS) {
    goto fail_kern;
  }

  if (clEnqueueNDRangeKernel(dst->q, k, 1, NULL, &bytes, NULL, 0, NULL, &ev) != CL_SUCCESS) {
    goto fail_kern;
  }
  
  if (clWaitForEvents(1, &ev) == CL_SUCCESS) {
    /* success! */
    res = 0;
  }
  
  clReleaseEvent(ev);
 fail_kern:
  clReleaseKernel(k);
 fail_prog:
  clReleaseProgram(p);
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
  return 0;
}

static const char *cl_error(void) {
  return get_error_string(err);
}

compyte_buffer_ops opencl_ops = {cl_alloc, cl_free, cl_move, cl_read, cl_write, cl_memset, cl_offset, cl_error};
