#ifndef _PYGPU_NDARRAY_H
#define _PYGPU_NDARRAY_H

//#include <Python.h>
//#include <structmember.h>
#include <numpy/arrayobject.h>
#include <gpu_ndarray.cuh>
#include <pygpu_ndarray_object.h>
#include <stdio.h>

#include <cublas.h>
#include <pygpu_language.h>

/*
 * Return a PyGpuNdArray whose 'nd' dimensions are all 0.
 * if nd==-1, it is not initialized.
 */
PyObject * PyGpuNdArray_New(int nd=-1);

/**
 * Return 1 for a PyGpuNdArrayObject otw 0
 */
int 
PyGpuNdArray_Check(const PyObject * ob);

/**
 * Return 1 for a PyGpuNdArrayObject otw 0
 */
int 
PyGpuNdArray_CheckExact(const PyObject * ob);

/**
 * Transfer the contents of numpy array `obj` to `self`.
 *
 * self is reallocated to have the correct dimensions if necessary.
 */
int PyGpuNdArray_CopyFromArray(PyGpuNdArrayObject * self, PyArrayObject*obj);

/**
 * [Re]allocate a PyGpuNdArrayObject with access to 'nd' dimensions.
 *
 * Note: This does not allocate storage for data.
 */
static
int PyGpuNdArray_set_nd(PyGpuNdArrayObject * self, const int nd)
{
    if (nd != PyGpuNdArray_NDIM(self))
    {
        if(0) fprintf(stderr, "PyGpuNdArray_set_nd: modif nd=%i to nd=%i\n", PyGpuNdArray_NDIM(self), nd);
    
        if (PyGpuNdArray_DIMS(self)){
            free(PyGpuNdArray_DIMS(self));
            PyGpuNdArray_DIMS(self) = NULL;
            PyGpuNdArray_NDIM(self) = -1;
        }
        if (PyGpuNdArray_STRIDES(self)){
            free(PyGpuNdArray_STRIDES(self));
            PyGpuNdArray_STRIDES(self) = NULL;
            PyGpuNdArray_NDIM(self) = -1;
        }
        if (nd == -1) return 0;

        PyGpuNdArray_DIMS(self) = (npy_intp*)malloc(nd*sizeof(npy_intp));
        if (NULL == PyGpuNdArray_DIMS(self))
        {
            PyErr_SetString(PyExc_MemoryError, "PyGpuNdArray_set_nd: Failed to allocate dimensions");
            return -1;
        }
        PyGpuNdArray_STRIDES(self) = (npy_intp*)malloc(nd*sizeof(npy_intp));
        if (NULL == PyGpuNdArray_STRIDES(self))
        {
            PyErr_SetString(PyExc_MemoryError, "PyGpuNdArray_set_nd: Failed to allocate str");
            return -1;
        }
        //initialize all dimensions and strides to 0
        for (int i = 0; i < nd; ++i)
        {
            PyGpuNdArray_DIM(self, i) = 0;
            PyGpuNdArray_STRIDES(self)[i] = 0;
        }

        PyGpuNdArray_NDIM(self) = nd;
	if(0) fprintf(stderr, "PyGpuNdArray_set_nd: end\n");
    }
    return 0;
}

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
    int verbose = 0;
    if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: start nd=%i\n descr=%p", nd);

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
    if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: before itemsize descr=%p elsize=%i\n", self->descr, self->descr->elsize);
    int elsize = PyGpuNdArray_ITEMSIZE((PyObject*)self);
    if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: set_nd! elsize=%i\n", nd,elsize;)
    if(order != NPY_FORTRANORDER){
      if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: NPY_CORDER\n");
      for (int i = nd-1; i >= 0; --i){
        PyGpuNdArray_STRIDE(self,i) = size * elsize;
        PyGpuNdArray_DIM(self,i) = dim[i];
        size = size * dim[i];
      }
    }else{
      if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: NPY_FORTRANORDER\n");
      size = dim[nd-1];
      PyGpuNdArray_STRIDE(self, 0) = elsize;
      PyGpuNdArray_DIM(self, nd-1) = dim[nd-1];
      for (int i = 1; i < nd; ++i){
        PyGpuNdArray_STRIDE(self, i) = PyGpuNdArray_STRIDE(self, i-1) * dim[i-1];
        PyGpuNdArray_DIM(self, nd-i-1) = dim[nd-i-1];
        size = size * dim[i];
      }
    }

    if (self->data_allocated != size)
    {
        if (device_free(PyGpuNdArray_DATA(self)))
        {
            // Does this ever happen??  Do we need to set data_allocated or data to 0?
            return -1;
        }
        assert(size>0);
	if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: will allocate for size=%d elements\n", size);

        PyGpuNdArray_DATA(self) = (char*)device_malloc(size * PyGpuNdArray_ITEMSIZE((PyObject *)self));
        if (!PyGpuNdArray_DATA(self))
        {
            PyGpuNdArray_set_nd(self,-1);
            self->data_allocated = 0;
            PyGpuNdArray_DATA(self) = 0;
            return -1;
        }

        self->data_allocated = size;
	self->gpu_ndarray.flags = NPY_DEFAULT;
	PyGpuNdArray_FLAGS(self) |= NPY_WRITEABLE;
	PyGpuNdArray_FLAGS(self) |= NPY_OWNDATA;
	if(nd == 0){
	  PyGpuNdArray_FLAGS(self) &= ~NPY_F_CONTIGUOUS;
	  PyGpuNdArray_FLAGS(self) |= NPY_C_CONTIGUOUS;
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
    }else{
      assert(0);// How to check for the flags? Need to check if already contiguous.
    }

    if (order != NPY_FORTRANORDER) {
        assert(PyGpuNdArray_is_c_contiguous(self));
    } else {
        assert(PyGpuNdArray_is_f_contiguous(self));
    }
    if (verbose) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: end\n");
    return 0;
}


#endif

/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  c-file-offsets:((innamespace . 0)(inline-open . 0))
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:encoding=utf-8:textwidth=79 :
