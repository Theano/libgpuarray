#ifndef _GPU_NDARRAY_H
#define _GPU_NDARRAY_H

//#include <Python.h>
//#include <structmember.h>
#include <numpy/arrayobject.h>
#include <gpu_ndarray.cuh>
#include <stdio.h>

#include <cublas.h>

#ifdef __DEVICE_EMULATION__
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     1  //This prevents printf from getting tangled up
#else
#define NUM_VECTOR_OP_BLOCKS                4096 //Max number of blocks to launch.  Should be read from device properties. (#10)
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     256  //Should be read from device properties. (#10)
#endif

#if 0
// Do not wait after every kernel & transfer.
#define CNDA_THREAD_SYNC
#else
// This is useful for using normal profiling tools
#define CNDA_THREAD_SYNC cudaThreadSynchronize();
#endif


#ifndef SHARED_SIZE 
#define SHARED_SIZE (16*1024)
#endif

/**
 * Allocation and freeing of device memory should go through these functions so that the lib can track memory usage.
 *
 * device_malloc will set the Python error message before returning None.
 * device_free will return nonzero on failure (after setting the python error message)
 */
void * device_malloc(size_t size);
int device_free(void * ptr);

/**
 * struct PyGPUArrayObject
 *
 * This is a Python type.  
 *
 */
typedef struct PyGpuNdArrayObject{
  PyObject_HEAD

  GpuNdArray gpu_ndarray; //no pointer, just inlined.
  PyObject * base;
  PyArray_Descr * descr; // for numpy-like desc
  int data_allocated; //the number of bytes allocated for devdata
} PyGpuNdArrayObject;

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

const npy_intp * 
PyGpuNdArray_DIMS(const PyGpuNdArrayObject * self)
{
    return self->gpu_ndarray.dimensions;
}
const npy_intp * 
PyGpuNdArray_STRIDES(const PyGpuNdArrayObject * self)
{
    return self->gpu_ndarray.strides;
}

#define PyGpuNdArray_NDIM(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.nd)
#define PyGpuNdArray_DATA(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.data)
#define PyGpuNdArray_BYTES(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.data)
#define PyGpuNdArray_OFFSET(obj) (((PyGpuNdArrayObject *)(obj))->gpu_ndarray.offset)
#define PyGpuNdArray_DIMS(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.dimensions)
#define PyGpuNdArray_STRIDES(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.strides)
#define PyGpuNdArray_DIM(obj,n) (PyGpuNdArray_DIMS(obj)[n])
#define PyGpuNdArray_STRIDE(obj,n) (PyGpuNdArray_STRIDES(obj)[n])
#define PyGpuNdArray_BASE(obj) (((PyGpuNdArrayObject *)obj)->base)
#define PyGpuNdArray_DESCR(obj) (((PyGpuNdArrayObject *)obj)->descr)
#define PyGpuNdArray_FLAGS(obj) (((PyGpuNdArrayObject *)obj)->gpu_ndarray.flags)
#define PyGpuNdArray_ITEMSIZE(obj) (((PyGpuNdArrayObject *)obj)->descr->elsize)
#define PyGpuNdArray_TYPE(obj) (((PyGpuNdArrayObject *)(obj))->descr->type_num)

#define PyGpuNdArray_SIZE(obj) PyArray_MultiplyList(PyGpuNdArray_DIMS(obj),PyGpuNdArray_NDIM(obj))
//npy_intp PyGpuNdArray_Size(PyObject* obj);
//npy_intp PyGpuNdArray_NBYTES(PyObject* arr);

/*
  Flags accessor
 */
#define PyGpuNdArray_CHKFLAGS(m, FLAGS)                              \
        ((((PyGpuNdArrayObject *)(m))->gpu_ndarray.flags & (FLAGS)) == (FLAGS))

#define PyGpuNdArray_ISCONTIGUOUS(m) PyGpuNdArray_CHKFLAGS(m, NPY_CONTIGUOUS)
#define PyGpuNdArray_ISFORTRAN(m) PyGpuNdArray_CHKFLAGS(m, NPY_F_CONTIGUOUS)
#define PyGpuNdArray_ISONESEGMENT(m) (PyGpuNdArray_ISCONTIGUOUS(m) || PyGpuNdArray_ISFORTRAN(m))
#define PyGpuNdArray_ISWRITEABLE(m) PyGpuNdArray_CHKFLAGS(m, NPY_WRITEABLE)
#define PyGpuNdArray_ISALIGNED(m) PyGpuNdArray_CHKFLAGS(m, NPY_ALIGNED)

#define PyGpuNdArray_ISNBO(arg) ((arg) != NPY_OPPBYTE)
#define PyGpuNdArray_IsNativeByteOrder PyArray_ISNBO
#define PyGpuNdArray_ISNOTSWAPPED(m) PyArray_ISNBO(PyArray_DESCR(m)->byteorder)
#define PyGpuNdArray_FLAGSWAP(m, flags) (PyGpuNdArray_CHKFLAGS(m, flags) && PyGpuNdArray_ISNOTSWAPPED(m))

#define PyGpuNdArray_ISCARRAY(m) PyGpuNdArray_FLAGSWAP(m, NPY_CARRAY)
#define PyGpuNdArray_ISCARRAY_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_CARRAY_RO)
#define PyGpuNdArray_ISFARRAY(m) PyGpuNdArray_FLAGSWAP(m, NPY_FARRAY)
#define PyGpuNdArray_ISFARRAY_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_FARRAY_RO)
#define PyGpuNdArray_ISBEHAVED(m) PyGpuNdArray_FLAGSWAP(m, NPY_BEHAVED)
#define PyGpuNdArray_ISBEHAVED_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_ALIGNED)


void PyGpuNdArray_fprint(FILE * fd, const PyGpuNdArrayObject *self)
{
    fprintf(fd, "PyGpuNdArrayObject <%p, %p> nd=%i data_allocated=%d\n",
	    self, PyGpuNdArray_DATA(self), PyGpuNdArray_NDIM(self), self->data_allocated);
    fprintf(fd, "\tHOST_DIMS:      ");
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
    {
        fprintf(fd, "%i\t", PyGpuNdArray_DIMS(self)[i]);
    }
    fprintf(fd, "\n\tHOST_STRIDES: ");
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
    {
        fprintf(fd, "%i\t", PyGpuNdArray_STRIDES(self)[i]);
    }
    fprintf(fd, "\n\tFLAGS: ");
    fprintf(fd, "\n\t\tC_CONTIGUOUS: %d", PyGpuNdArray_ISCONTIGUOUS(self));
    fprintf(fd, "\n\t\tF_CONTIGUOUS: %d", PyGpuNdArray_ISFORTRAN(self));
    fprintf(fd, "\n\t\tOWNDATA: %d", PyGpuNdArray_CHKFLAGS(self, NPY_OWNDATA));
    fprintf(fd, "\n\t\tWRITEABLE: %d", PyGpuNdArray_ISWRITEABLE(self));
    fprintf(fd, "\n\t\tALIGNED: %d", PyGpuNdArray_ISALIGNED(self));
    fprintf(fd, "\n\t\tUPDATEIFCOPY: %d", PyGpuNdArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    fprintf(fd, "\n");

}
void PyArray_fprint(FILE * fd, const PyArrayObject *self)
{
    fprintf(fd, "PyArrayObject <%p, %p> nd=%i\n",
	    self, PyArray_DATA(self), PyArray_NDIM(self));
    fprintf(fd, "\tHOST_DIMS:      ");
    for (int i = 0; i < PyArray_NDIM(self); ++i)
    {
        fprintf(fd, "%i\t", PyArray_DIMS(self)[i]);
    }
    fprintf(fd, "\n\tHOST_STRIDES: ");
    for (int i = 0; i < PyArray_NDIM(self); ++i)
    {
        fprintf(fd, "%i\t", PyArray_STRIDES(self)[i]);
    }
    fprintf(fd, "\n\tFLAGS: ");
    fprintf(fd, "\n\t\tC_CONTIGUOUS: %d", PyArray_ISCONTIGUOUS(self));
    fprintf(fd, "\n\t\tF_CONTIGUOUS: %d", PyArray_ISFORTRAN(self));
    fprintf(fd, "\n\t\tOWNDATA: %d", PyArray_CHKFLAGS(self, NPY_OWNDATA));
    fprintf(fd, "\n\t\tWRITEABLE: %d", PyArray_ISWRITEABLE(self));
    fprintf(fd, "\n\t\tALIGNED: %d", PyArray_ISALIGNED(self));
    fprintf(fd, "\n\t\tUPDATEIFCOPY: %d", PyArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    fprintf(fd, "\n");

}


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
    if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: start nd=%i\n descr=%p", nd);
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
    if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: before itemsize descr=%p elsize=%i\n", self->descr, self->descr->elsize);
    int elsize = PyGpuNdArray_ITEMSIZE((PyObject*)self);
    if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: set_nd! elsize=%i\n", nd,elsize);
    if(order != NPY_FORTRANORDER){
      if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: NPY_CORDER\n");
      for (int i = nd-1; i >= 0; --i){
        PyGpuNdArray_STRIDE(self,i) = size * elsize;
        PyGpuNdArray_DIM(self,i) = dim[i];
        size = size * dim[i];
      }
    }else{
      if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: NPY_FORTRANORDER\n");
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
	if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: will allocate for size=%d elements\n", size);

        PyGpuNdArray_DATA(self) = (char*)device_malloc(size * PyGpuNdArray_ITEMSIZE((PyObject *)self));
        if (!PyGpuNdArray_DATA(self))
        {
            PyGpuNdArray_set_nd(self,-1);
            self->data_allocated = 0;
            PyGpuNdArray_DATA(self) = 0;
            return -1;
        }
        if (0)
            fprintf(stderr,
                "Allocated data %p (self=%p)\n",
                PyGpuNdArray_DATA(self),
                self);
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
    if(0) fprintf(stderr, "PyGpuNdArray_alloc_contiguous: end\n");
    return 0;
}




template <typename T>
static T ceil_intdiv(T a, T b)
{
    return (a/b) + ((a % b) ? 1: 0);
}

#endif
