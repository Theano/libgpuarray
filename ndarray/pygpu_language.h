/**
 * This file contain the header for ALL code that depend on cuda or opencl.
 */
#ifndef _PYGPU_LANGUAGE_H
#define _PYGPU_LANGUAGE_H
#include <Python.h>
//#include <iostream>

#include "pygpu_ndarray_object.h"

/////////////////////////
// Alloc and Free
/////////////////////////
//If true, when there is a gpu malloc or free error, we print the size of allocated memory on the device.
#define COMPUTE_GPU_MEM_USED 0
#define VERBOSE_ALLOC_FREE 0
//If true, we fill with NAN allocated device memory.
#define ALLOC_MEMSET 0

static int _outstanding_mallocs[] = {0,0};

#ifdef DEBUG
#define DPRINTF(args...) fprintf(stderr, args)
#else
#define DPRINTF(...)
#endif

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

int PyGpuNdArray_CopyFromPyGpuNdArray(PyGpuNdArrayObject * self, PyGpuNdArrayObject * other, bool unbroadcast = false);

/**
 * PyGpuNdArray_alloc_contiguous
 *
 * Allocate storage space for a tensor of rank 'nd' and given dimensions.
 *
 * Note: PyGpuNdArray_alloc_contiguous is templated to work for both int dimensions and npy_intp dimensions
 */
template<typename inttype>
int PyGpuNdArray_alloc_contiguous(PyGpuNdArrayObject *self, const int nd, const inttype * dim, NPY_ORDER order=NPY_CORDER)
{
    DPRINTF("PyGpuNdArray_alloc_contiguous: start nd=%i descr=%p\n", nd, self);

    if (!PyGpuNdArray_DESCR(self)){
        PyErr_SetString(PyExc_ValueError,
                        "PyGpuNdArray_alloc_contiguous: The array don't have a type! We can't allocate it!\n");
        return -1;
    }
    
    // allocate an empty ndarray with c_contiguous access
    // return 0 on success
    int size = 1; //set up the strides for contiguous tensor
    assert (nd >= 0);
    if (PyGpuNdArray_set_nd(self, nd))
    {
        return -1;
    }
    //TODO: check if by any chance our current dims are correct,
    //      and strides already contiguous
    //      in that case we can return right here.
    DPRINTF("PyGpuNdArray_alloc_contiguous: before itemsize descr=%p elsize=%i\n", self->descr, self->descr->elsize);
    int elsize = PyGpuNdArray_ITEMSIZE((PyObject*)self);
    DPRINTF("PyGpuNdArray_alloc_contiguous: set_nd %d! elsize=%i\n", nd, elsize);
    if(order != NPY_FORTRANORDER){
      DPRINTF("PyGpuNdArray_alloc_contiguous: NPY_CORDER\n");
      for (int i = nd-1; i >= 0; --i){
	if (size == 0)
	  PyGpuNdArray_STRIDE(self, i) = elsize;
	else
	  PyGpuNdArray_STRIDE(self,i) = size * elsize;
        PyGpuNdArray_DIM(self,i) = dim[i];
        size = size * dim[i];
      }
    }else if (nd>0){
      DPRINTF("PyGpuNdArray_alloc_contiguous: NPY_FORTRANORDER\n");
      size = dim[0];
      PyGpuNdArray_STRIDE(self, 0) = elsize;
      PyGpuNdArray_DIM(self, nd-1) = dim[nd-1];
      for (int i = 1; i < nd; ++i){
	if (size == 0)
	  PyGpuNdArray_STRIDE(self, i) = elsize;
	else
	  PyGpuNdArray_STRIDE(self, i) = PyGpuNdArray_STRIDE(self, i-1) * dim[i-1];
        PyGpuNdArray_DIM(self, nd-i-1) = dim[nd-i-1];
        size = size * dim[i];
      }
    }

    if (self->data_allocated != size)
    {
        // If self is a view, do not try to free its memory
        if (self->data_allocated && device_free(PyGpuNdArray_DATA(self))) {
	  // Does this ever happen??  Do we need to set data_allocated or devdata to 0?
	  PyGpuNdArray_DATA(self) = NULL;
	  self->data_allocated = 0;
	  return -1;
	}

        assert(size>0);
	DPRINTF("PyGpuNdArray_alloc_contiguous: will allocate for size=%d elements\n", size);

        PyGpuNdArray_DATA(self) = (char*)device_malloc(size * PyGpuNdArray_ITEMSIZE((PyObject *)self));
        if (!PyGpuNdArray_DATA(self))
        {
            PyGpuNdArray_set_nd(self,-1);
            self->data_allocated = 0;
            PyGpuNdArray_DATA(self) = 0;
            return -1;
        }

	// The structure of self will be reused with newly allocated memory.
	// If self was a view, we should remove the reference to its base.
	// (If base was already NULL, the following has no effect.)
	Py_XDECREF(self->base);
	self->base = NULL;

        self->data_allocated = size;
	self->gpu_ndarray.flags = NPY_DEFAULT;
	PyGpuNdArray_FLAGS(self) |= NPY_WRITEABLE;
	PyGpuNdArray_FLAGS(self) |= NPY_OWNDATA;
	if (nd == 0) {
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	  if (order != NPY_FORTRANORDER) {
	    PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
	  } else {
	    PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  }

	}else if(nd == 1){//set c and f contiguous
	  PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	}else if(order != NPY_FORTRANORDER){//set c contiguous
	  PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	}else{//set f contiguous
	  PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) &= ~NPY_C_CONTIGUOUS;
	}
	PyGpuNdArray_FLAGS(self) &= ~NPY_UPDATEIFCOPY;
    }else if(size == 0){
      PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
      PyGpuNdArray_FLAGS(self) |= NPY_OWNDATA;
	if (nd == 0) {
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	  if (order != NPY_FORTRANORDER) {
	    PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
	  } else {
	    PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  }

	}else if(nd == 1){//set c and f contiguous
	  PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	}else if(order != NPY_FORTRANORDER){//set c contiguous
	  PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
	}else{//set f contiguous
	  PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) &= ~NPY_C_CONTIGUOUS;
	}
	PyGpuNdArray_FLAGS(self) &= ~NPY_UPDATEIFCOPY;
        return 0;
    }else{
      // How to check for the flags? Need to check if already contiguous.
      PyErr_Format(PyExc_RuntimeError,
		   "PyGpuNdArray_alloc_contiguous: self->data_allocated=%d, size=%d, cmp=%d",
		   self->data_allocated, size, self->data_allocated != size
		   );
      return -1;
    }

    if (order != NPY_FORTRANORDER) {
        assert(PyGpuNdArray_is_c_contiguous(self));
    } else {
        assert(PyGpuNdArray_is_f_contiguous(self));
    }
    DPRINTF("PyGpuNdArray_alloc_contiguous: end\n");
    return 0;
}

enum PyGpuTransfert { PyGpuHostToDevice, PyGpuDeviceToHost };
int PyGpuMemcpy(void * dst, const void * src, int dev_offset, size_t bytes, PyGpuTransfert direction);

int PyGpuMemset(void * dst, int data, size_t bytes);
#endif
