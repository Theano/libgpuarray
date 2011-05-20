/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef _PYGPU_LANGUAGE_H
#define _PYGPU_LANGUAGE_H
#include <Python.h>
//#include <iostream>
//#include <pygpu_ndarray.cuh>

/////////////////////////
// Alloc and Free
/////////////////////////
//If true, when there is a gpu malloc or free error, we print the size of allocated memory on the device.
#define COMPUTE_GPU_MEM_USED 0
#define VERBOSE_ALLOC_FREE 0
//If true, we fill with NAN allocated device memory.
#define ALLOC_MEMSET 0

static int _outstanding_mallocs[] = {0,0};

#if COMPUTE_GPU_MEM_USED
int _allocated_size = 0;
const int TABLE_SIZE = 10000;
struct table_struct{
  void* ptr;
  int size;
};
table_struct _alloc_size_table[TABLE_SIZE];
#endif

/**
 * Allocation and freeing of device memory should go through these functions so that the lib can track memory usage.
 *
 * device_malloc will set the Python error message before returning None.
 * device_free will return nonzero on failure (after setting the python error message)
 */
void * device_malloc(size_t size);
int device_free(void * ptr);
static PyObject *
outstanding_mallocs(PyObject* self, PyObject * args)
{
    return PyInt_FromLong(_outstanding_mallocs[0]);
}
/*
/////////////////////////////
// Satisfying reqs to be Type
/////////////////////////////

//DON'T use directly(if their is other PyGpuNdArrayObject that point to it, it will cause problem)! use Py_DECREF() instead
static void
PyGpuNdArrayObject_dealloc(PyGpuNdArrayObject* self)
{
    if (0) fprintf(stderr, "PyGpuNdArrayObject_dealloc\n");
    if (0) std::cerr << "PyGpuNdArrayObject dealloc " << self << '\n';
    if (0) std::cerr << "PyGpuNdArrayObject dealloc " << self << " " << PyGpuNdArray_DATA(self) << '\n';

    if(self->ob_refcnt>1)
      printf("WARNING:PyGpuNdArrayObject_dealloc called when their is still active reference to it.\n");

    if (self->data_allocated){
        assert(PyGpuNdArray_DATA(self));
        if (PyGpuNdArray_DATA(self)){
            if (device_free(PyGpuNdArray_DATA(self))){
	      fprintf(stderr,
		  "!!!! error freeing device memory %p (self=%p)\n",
		  PyGpuNdArray_DATA(self), self);
	    }
	    PyGpuNdArray_DATA(self) = NULL;
	}
    }
    PyGpuNdArray_OFFSET(self) = 0;
    PyGpuNdArray_NDIM(self) = -1;
    Py_XDECREF(self->base);
    self->base = NULL;
    if (PyGpuNdArray_DIMS(self)){
        free(PyGpuNdArray_DIMS(self));
        PyGpuNdArray_DIMS(self) = NULL;
    }
    if (PyGpuNdArray_STRIDES(self)){
        free(PyGpuNdArray_STRIDES(self));
        PyGpuNdArray_STRIDES(self) = NULL;
    }
    PyGpuNdArray_FLAGS(self) = NPY_DEFAULT;
    Py_XDECREF(self->descr);//TODO: How to handle the refcont on this object?
    self->descr = NULL;
    self->data_allocated = 0;

    self->ob_type->tp_free((PyObject*)self);
    --_outstanding_mallocs[1];
    if(0){
        fprintf(stderr, "device_malloc_counts: (device) %i (obj) %i\n",
                _outstanding_mallocs[0],
                _outstanding_mallocs[1]);
    }
}
*/
#endif
