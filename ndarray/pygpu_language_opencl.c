#include <sys/types.h>
#include <assert.h>
#include <stdio.h>

#include <pygpu_ndarray_object.h>
#include <pygpu_language.h>

#include <opencl.h>

cl_context ctx = NULL;
cl_device_id dev;
cl_command_queue q;

void
init_context(void)
{
  cl_int err;
  cl_program prog;
  size_t sz;

  if (ctx != NULL) return;

  ctx = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &err);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "Could not create context, will fail later");
    /* error - error - error */
    /* but we do nothing */
    return;
  }

  /* Maybe this will break with more than one device ... meh */
  if (clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 1, &dev, NULL) != CL_SUCCESS) goto fail;
  
  q = clCreateCommandQueue(ctx, dev, NULL, &err);
  if (err != CL_SUCCESS) goto fail;
  
  return;
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

  res = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "Could not allocate device memory");
    return NULL;
  }
  return res;
}

int
device_free(void * ptr)
{
  if (clReleaseMemObject((cl_mem)ptr) != CL_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "Could not free device memory");
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
  size_t size_source = 1;
  cl_event ev;
  
  assert(PyGpuNdArray_OFFSET(self) == 0 && PyGpuNdArray_OFFSET(other) == 0);
  assert(PyGpuNdArray_TYPE(self) == PyGpuNdArray_TYPE(other));
  assert(PyGpuNdArray_ISWRITEABLE(self));
  if (PyGpuNdArray_NDIM(self) == -1) {
    PyErr_SetString(PyExc_TypeError, "can't copy into un-initialized PyGpuN\
dArrayObject");
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
    size_source *= (unsigned int) PyGpuNdArray_DIMS(other)[i];
  }
  
  if (0 == size) {
    return 0; //nothing to copy, we're done.                                
  }
  
  if (clEnqueueCopyBuffer(q, (cl_mem)PyGpuNdArray_DATA(self),
			  (cl_mem)PyGpuNdArray_DATA(other),
			  0, 0, size, 0, NULL, &ev) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could create copy command");
    return -1;
  }
  if (clWaitForEvents(1, &ev) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not copy data");
    clReleaseEvent(ev);
    return -1;
  }
  clReleaseEvent(ev);

  return 0;
}

int
PyGpuMemcpy(void * dst, const void * src, size_t bytes,
	    PyGpuTransfert direction)
{
  cl_int err;
  cl_event ev;
  
  switch (direction)
    {
    case PyGpuHostToDevice:
      err = clEnqueueWriteBuffer(q, (cl_mem)dst, CL_FALSE, 0, bytes, src, 
				 0, NULL, &ev);
    case PyGpuDeviceToHost:
      err = clEnqueueReadBuffer(q, (cl_mem)src, CL_FALSE, 0, bytes, dst,
			       0, NULL, &ev);
    default:
      PyErr_Format(PyExc_ValueError, "Unknown direction %d", direction);
      return -1;
    }

  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create memcpy command");
    return -1;
  }

  if (clWaitForEvents(1, &ev) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not memcpy data");
    clReleaseEvent(ev);
    return -1;
  }
  clReleaseEvent(ev);

  return 0;
}

int
PyGpuMemSet(void * dst, int data, size_t bytes)
{ 
  /* This should be at least one byte over the formatted string below */
  char local_kern[92];
  size_t sz, local;
  int r, res = -1;

  cl_int err;
  cl_event ev;
  cl_program p;
  cl_kernel k;

  /* OpenCL devices do not always support byte-addressable storage
     and stuff will break when used in this way */
  assert(bytes % 4 == 0);
  bytes /= 4;

  unsigned char val = (unsigned)data;
  unsigned int pattern = (unsigned int)val & (unsigned int)val >> 8 & (unsigned int)val >> 16 & (unsigned int)val >> 24;

  r = snprintf(local_kern, sizeof(local_kern),
	       "__kernel void memset(__global unsigned int *mem) { mem[get_global_id(0)] = %u; }", pattern);
  /* If this assert fires, increase the size of local_kern above. */ 
  assert(r >= sizeof(local_kern));

  sz = strlen(local_kern);
  p = clCreateProgramWithSource(ctx, 1, (const char **)&local_kern, &sz, &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create program");
    return -1;
  }

  if (clBuildProgram(p, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not build program");
    goto fail_prog;
  }

  k = clCreateKernel(p, "memset", &err);
  if (err != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not create kernel");
    goto fail_prog;
  }

  if (clSetKernelArg(k, 0, sizeof(cl_mem), dst) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not set kernel arg");
    goto fail_kern;
  }

  if (clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE,
			       sizeof(local), &local, NULL) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not get workgroup info");
    goto fail_kern;
  }
  
  if (clEnqueueNDRangeKernel(q, k, 1, NULL, &bytes, &local, 0, NULL, &ev) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not enqueue kernel");
    goto fail_kern;
  }
  
  if (clWaitForEvents(1, &ev) != CL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "Could not memset");
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
