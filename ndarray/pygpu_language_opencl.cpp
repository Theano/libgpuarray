#include <sys/types.h>
#include <assert.h>
#include <stdio.h>

#include <pygpu_ndarray_object.h>
#include <pygpu_language.h>

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else

#include <CL/opencl.h>

#endif

cl_context ctx = NULL;
cl_device_id dev;
cl_command_queue q;

void setup_context(cl_context c);

static void
init_context(void)
{
  cl_int err;
  cl_uint n;
  cl_platform_id *plats;
  cl_context_properties props[3];
  cl_context c;

  if (ctx != NULL) return;

  err = clGetPlatformIDs(0, NULL, &n);
  if (err != CL_SUCCESS) return;

  plats = (cl_platform_id *)calloc(n, sizeof(cl_platform_id));
  if (plats == NULL) return;

  err = clGetPlatformIDs(n, plats, NULL);
  if (err != CL_SUCCESS) goto fail_id;

  props[0] = CL_CONTEXT_PLATFORM;
  props[1] = (cl_context_properties)plats[0];
  props[2] = 0;

  c = clCreateContextFromType(props, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Could not create context, will fail later (%d)!\n", err);
    /* error - error - error */
    /* but we do nothing */
    goto fail_id;
  }

  free(plats);

  setup_context(c);
  clReleaseContext(c);

  return;
 fail_id:
  free(plats);
}

void
setup_context(cl_context c) {
  cl_int err;
  cl_device_id *devs;
  size_t sz;

  if (ctx != NULL) {
    clReleaseContext(ctx);
    clReleaseCommandQueue(q);
  }
  ctx = c;
  clRetainContext(ctx);

  err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &sz);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clGetContextInfo = %d\n", err);
    goto fail;
  }

  devs = (cl_device_id *)malloc(sz);
  if (devs == NULL) goto fail;

  err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, sz, devs, NULL);
  if (err != CL_SUCCESS) goto fail_dev;

  dev = devs[0];
  free(devs);

  q = clCreateCommandQueue(ctx, dev, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clCreateCommandQueue = %d", err);
    goto fail;
  }

  return;
 fail_dev:
  free(devs);
 fail:
  clReleaseContext(ctx);
  ctx = NULL;
}

void *
device_malloc(size_t size)
{
  cl_int err;
  cl_mem res;

  init_context();

  DPRINTF("malloc size = %zu\n", size);

  /* OpenCL devices do not always support byte-addressable storage
     therefore make sure we have at least 4 bytes in buffers */
  if (size < 4) size = 4;

  res = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "Could not allocate device memory (%d)", err);
    return NULL;
  }

  return res;
}

int
device_free(void * ptr)
{
  cl_int err;

  if ((err = clReleaseMemObject((cl_mem)ptr)) != CL_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "Could not free device memory (%d)", err);
    return -1;
  }
  return 0;
}

int
PyGpuNdArray_CopyFromPyGpuNdArray(PyGpuNdArrayObject * self,
				  PyGpuNdArrayObject * other,
				  bool unbroadcast)
{
  size_t size = 1;
  cl_event ev;
  cl_int err;
  
  assert(PyGpuNdArray_TYPE(self) == PyGpuNdArray_TYPE(other));
  assert(PyGpuNdArray_ISWRITEABLE(self));
  if (PyGpuNdArray_NDIM(self) == -1) {
    PyErr_SetString(PyExc_TypeError, "can't copy into un-initialized PyGpuN\
dArrayObject");
    return -1;
  }

  if (!(PyGpuNdArray_ISONESEGMENT(self) && PyGpuNdArray_ISONESEGMENT(other))) {
    PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: only contiguous arrays are supported");
    return -1;
  }

  if ((PyGpuNdArray_ISCONTIGUOUS(self) != PyGpuNdArray_ISCONTIGUOUS(other)) ||
      (PyGpuNdArray_ISFORTRAN(self) != PyGpuNdArray_ISFORTRAN(other))
      ) {
    PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: the input and output don't have the same c/f contiguous memory layout. This isnot supported now.");
    return -1;
  }

  if (PyGpuNdArray_NDIM(self) != PyGpuNdArray_NDIM(other)) {
    PyErr_Format(PyExc_NotImplementedError, "PyGpuNdArray_CopyFromPyGpuNdArray: need same number of dims. destination nd=%d, source nd=%d. No broadcasting implemented.", PyGpuNdArray_NDIM(self), PyGpuNdArray_NDIM(other));
    return -1;
  }

  for (int i = 0; i< PyGpuNdArray_NDIM(self); ++i) {
    if ((PyGpuNdArray_DIMS(self)[i] != PyGpuNdArray_DIMS(other)[i])
	&& (1!=PyGpuNdArray_DIMS(other)[i] || !unbroadcast) ) {
      PyErr_Format(PyExc_ValueError, "need same dimensions for dim %d, destination=%ld, source=%ld",
		   i, PyGpuNdArray_DIMS(self)[i], PyGpuNdArray_DIMS(other)[i]);
      return -1;
    }
    size *= (unsigned int) PyGpuNdArray_DIMS(self)[i];
  }

  if (0 == size) {
    return 0; //nothing to copy, we're done.
  }
  size *= PyGpuNdArray_ITEMSIZE(self);

  if ((err = clEnqueueCopyBuffer(q, (cl_mem)PyGpuNdArray_DATA(other),
				 (cl_mem)PyGpuNdArray_DATA(self),
				 PyGpuNdArray_OFFSET(other),
				 PyGpuNdArray_OFFSET(self),
				 size, 0, NULL, &ev)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create copy command (%d)", err);
    return -1;
  }
  if ((err = clWaitForEvents(1, &ev)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not copy data (%d)", err);
    clReleaseEvent(ev);
    return -1;
  }
  clReleaseEvent(ev);

  return 0;
}

int
PyGpuMemcpy(void * dst, const void * src, int dev_offset, size_t bytes,
	    PyGpuTransfert direction)
{
  cl_int err;
  cl_event ev;
  
  switch (direction)
    {
    case PyGpuHostToDevice:
      err = clEnqueueWriteBuffer(q, (cl_mem)dst, CL_FALSE, dev_offset, bytes,
				 src, 0, NULL, &ev);
      break;
    case PyGpuDeviceToHost:
      err = clEnqueueReadBuffer(q, (cl_mem)src, CL_FALSE, dev_offset, bytes,
				dst, 0, NULL, &ev);
      break;
    default:
      PyErr_Format(PyExc_ValueError, "Unknown direction %d", direction);
      return -1;
    }

  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create memcpy command (%d)", err);
    return -1;
  }

  if ((err = clWaitForEvents(1, &ev)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not memcpy data (%d)", err);
    clReleaseEvent(ev);
    return -1;
  }
  clReleaseEvent(ev);

  return 0;
}

int
PyGpuMemset(void * dst, int data, size_t bytes)
{ 
  /* This should be at least one byte over the formatted string below */
  char local_kern[92];
  const char *rlk[1];
  size_t sz;
  int r, res = -1;

  cl_int err;
  cl_event ev;
  cl_program p;
  cl_kernel k;

  bytes = (bytes+3)/4;

  if (bytes == 0)
    return 0;

  unsigned char val = (unsigned)data;
  unsigned int pattern = (unsigned int)val & (unsigned int)val >> 8 & (unsigned int)val >> 16 & (unsigned int)val >> 24;

  r = snprintf(local_kern, sizeof(local_kern),
	       "__kernel void memset(__global unsigned int *mem) { mem[get_global_id(0)] = %u; }", pattern);
  /* If this assert fires, increase the size of local_kern above. */ 
  assert(r >= sizeof(local_kern));


  sz = strlen(local_kern);
  rlk[0] = local_kern;
  p = clCreateProgramWithSource(ctx, 1, rlk, &sz, &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create program (%d)", err);
    return -1;
  }

  if ((err = clBuildProgram(p, 1, &dev, NULL, NULL, NULL)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not build program (%d)", err);
    goto fail_prog;
  }

  k = clCreateKernel(p, "memset", &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create kernel (%d)", err);
    goto fail_prog;
  }

  if ((err = clSetKernelArg(k, 0, sizeof(cl_mem), &dst)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not set kernel arg (%d)", err);
    goto fail_kern;
  }

  if ((err = clEnqueueNDRangeKernel(q, k, 1, NULL, &bytes, NULL, 0, NULL, &ev)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not enqueue kernel (%d)", err);
    goto fail_kern;
  }
  
  if ((err = clWaitForEvents(1, &ev)) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not memset (%d)", err);
  }
  
  /* success! */
  res = 0;
  
  clReleaseEvent(ev);
 fail_kern:
  clReleaseKernel(k);
 fail_prog:
  clReleaseProgram(p);
  return res;
}
