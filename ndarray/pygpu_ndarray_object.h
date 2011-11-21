/**
 * struct PyGPUArrayObject
 *
 * This is a Python type.  
 *
 */
#ifndef _PYGPU_NDARRAY_OBJECT_H
#define _PYGPU_NDARRAY_OBJECT_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include "gpu_ndarray.h"

typedef struct PyGpuNdArrayObject{
  PyObject_HEAD

  GpuNdArray gpu_ndarray; //no pointer, just inlined.
  PyObject * base;
  PyArray_Descr * descr; // for numpy-like desc
  int data_allocated; //the number of bytes allocated for devdata
} PyGpuNdArrayObject;

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
#define PyGpuNdArray_ISFORTRAN(m) (PyGpuNdArray_CHKFLAGS(m, NPY_F_CONTIGUOUS) && \
                                   PyGpuNdArray_NDIM(m) > 1)
#define PyGpuNdArray_FORTRAN_IF(m) (PyGpuNdArray_CHKFLAGS(m, NPY_F_CONTIGUOUS)? \
                                    NPY_F_CONTIGUOUS : 0)
#define PyGpuNdArray_ISONESEGMENT(m) (PyGpuNdArray_NDIM(m) == 0 || \
                                      PyGpuNdArray_ISCONTIGUOUS(m) || \
                                      PyGpuNdArray_ISFORTRAN(m))
#define PyGpuNdArray_ISWRITEABLE(m) PyGpuNdArray_CHKFLAGS(m, NPY_WRITEABLE)
#define PyGpuNdArray_ISALIGNED(m) PyGpuNdArray_CHKFLAGS(m, NPY_ALIGNED)

#define PyGpuNdArray_ISNBO(arg) ((arg) != NPY_OPPBYTE)
// THE NEXT ONE SEEM BAD...
#define PyGpuNdArray_IsNativeByteOrder PyArray_ISNBO
#define PyGpuNdArray_ISNOTSWAPPED(m) PyArray_ISNBO(PyArray_DESCR(m)->byteorder)
#define PyGpuNdArray_FLAGSWAP(m, flags) (PyGpuNdArray_CHKFLAGS(m, flags) && PyGpuNdArray_ISNOTSWAPPED(m))

#define PyGpuNdArray_ISCARRAY(m) PyGpuNdArray_FLAGSWAP(m, NPY_CARRAY)
#define PyGpuNdArray_ISCARRAY_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_CARRAY_RO)
#define PyGpuNdArray_ISFARRAY(m) PyGpuNdArray_FLAGSWAP(m, NPY_FARRAY)
#define PyGpuNdArray_ISFARRAY_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_FARRAY_RO)
#define PyGpuNdArray_ISBEHAVED(m) PyGpuNdArray_FLAGSWAP(m, NPY_BEHAVED)
#define PyGpuNdArray_ISBEHAVED_RO(m) PyGpuNdArray_FLAGSWAP(m, NPY_ALIGNED)

static
void PyGpuNdArray_fprint(FILE * fd, const PyGpuNdArrayObject *self)
{
    fprintf(fd, "PyGpuNdArrayObject <%p, %p> nd=%i data_allocated=%d\n",
	    self, PyGpuNdArray_DATA(self), PyGpuNdArray_NDIM(self), self->data_allocated);
    fprintf(fd, "\tITEMSIZE: %d\n", PyGpuNdArray_ITEMSIZE(self));
    fprintf(fd, "\tTYPENUM: %d\n", PyGpuNdArray_TYPE(self));
    fprintf(fd, "\tRefcount: %ld\n", (long int)self->ob_refcnt);
    fprintf(fd, "\tBASE: %p\n", PyGpuNdArray_BASE(self));
    fprintf(fd, "\tHOST_DIMS:      ");
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
    {
        fprintf(fd, "%ld\t", PyGpuNdArray_DIMS(self)[i]);
    }
    fprintf(fd, "\n\tHOST_STRIDES: ");
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
    {
        fprintf(fd, "%ld\t", PyGpuNdArray_STRIDES(self)[i]);
    }
    fprintf(fd, "\n\tFLAGS: ");
    fprintf(fd, "\n\t\tC_CONTIGUOUS: %d", PyGpuNdArray_ISCONTIGUOUS(self));
    fprintf(fd, "\n\t\tPyGpuNdArray_ISFORTRAN: %d PyGpuNdArray_FORTRAN_IF:%d F_CONTIGUOUS: %d",
            PyGpuNdArray_ISFORTRAN(self), PyGpuNdArray_FORTRAN_IF(self), PyGpuNdArray_CHKFLAGS(self, NPY_FORTRAN));
    fprintf(fd, "\n\t\tOWNDATA: %d", PyGpuNdArray_CHKFLAGS(self, NPY_OWNDATA));
    fprintf(fd, "\n\t\tWRITEABLE: %d", PyGpuNdArray_ISWRITEABLE(self));
    fprintf(fd, "\n\t\tALIGNED: %d", PyGpuNdArray_ISALIGNED(self));
    fprintf(fd, "\n\t\tUPDATEIFCOPY: %d", PyGpuNdArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    fprintf(fd, "\n");

}
static
void PyArray_fprint(FILE * fd, const PyArrayObject *self)
{
    fprintf(fd, "PyArrayObject <%p, %p> nd=%i\n",
	    self, PyArray_DATA(self), PyArray_NDIM(self));
    fprintf(fd, "\tITEMSIZE: %d\n", PyArray_ITEMSIZE(self));
    fprintf(fd, "\tTYPENUM: %d\n", PyArray_TYPE(self));
    fprintf(fd, "\tHOST_DIMS:      ");
    for (int i = 0; i < PyArray_NDIM(self); ++i)
    {
        fprintf(fd, "%ld\t", PyArray_DIMS(self)[i]);
    }
    fprintf(fd, "\n\tHOST_STRIDES: ");
    for (int i = 0; i < PyArray_NDIM(self); ++i)
    {
        fprintf(fd, "%ld\t", PyArray_STRIDES(self)[i]);
    }
    fprintf(fd, "\n\tFLAGS: ");
    fprintf(fd, "\n\t\tC_CONTIGUOUS: %d", PyArray_ISCONTIGUOUS(self));
    fprintf(fd, "\n\t\tPyArray_ISFORTRAN: %d PyArray_FORTRAN_IF:%d F_CONTIGUOUS: %d",
            PyArray_ISFORTRAN(self), PyArray_FORTRAN_IF(self), PyArray_CHKFLAGS(self, NPY_FORTRAN));
    fprintf(fd, "\n\t\tOWNDATA: %d", PyArray_CHKFLAGS(self, NPY_OWNDATA));
    fprintf(fd, "\n\t\tWRITEABLE: %d", PyArray_ISWRITEABLE(self));
    fprintf(fd, "\n\t\tALIGNED: %d", PyArray_ISALIGNED(self));
    fprintf(fd, "\n\t\tUPDATEIFCOPY: %d", PyArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    fprintf(fd, "\n");

}

template <typename T>
static T ceil_intdiv(T a, T b)
{
    return (a/b) + ((a % b) ? 1: 0);
}


//Compute if the resulting array is c contiguous
static bool
PyGpuNdArray_is_c_contiguous(const PyGpuNdArrayObject * self)
{
    bool c_contiguous = true;
    int size = PyGpuNdArray_ITEMSIZE(self);
    for (int i = PyGpuNdArray_NDIM(self)-1; (i >= 0) && c_contiguous; --i) {
        if (PyGpuNdArray_STRIDE(self, i) != size) {
            c_contiguous = false;
        }
        size = size * PyGpuNdArray_DIM(self, i);
    }
    return c_contiguous;
}

//Compute if the resulting array is f contiguous
static bool
PyGpuNdArray_is_f_contiguous(const PyGpuNdArrayObject * self)
{
    bool f_contiguous = true;
    int size = PyGpuNdArray_ITEMSIZE(self);
    for (int i = 0; i < PyGpuNdArray_NDIM(self) && f_contiguous; ++i) {
        if (PyGpuNdArray_STRIDE(self, i) != size) {
            f_contiguous = false;
        }
        size = size * PyGpuNdArray_DIM(self, i);
    }
    return f_contiguous;
}

static PyObject *
PyGpuNdArray_as_c_contiguous(PyObject* dummy, PyObject* args, PyObject *kargs);
static PyObject *
PyGpuNdArray_as_f_contiguous(PyObject* dummy, PyObject* args, PyObject *kargs);

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
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
