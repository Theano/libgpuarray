#ifndef _PYGPU_NDARRAY_H
#define _PYGPU_NDARRAY_H
#ifndef OFFSET
#define OFFSET 0
#endif

//#include <Python.h>
//#include <structmember.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

#include "pygpu_ndarray_object.h"
#include "gpu_ndarray.h"
#include "pygpu_language.h"

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

static int 
PyGpuNdArray_add_offset(PyGpuNdArrayObject * self, int offset);

static int 
PyGpuNdArray_set_data(PyGpuNdArrayObject * self, char * data, PyObject * base, int offset=0);

static PyObject *
PyGpuNdArray_Subscript(PyObject * py_self, PyObject * key);

static PyObject *
PyGpuNdArray_Copy(PyGpuNdArrayObject * self, NPY_ORDER order=NPY_CORDER);

static PyObject *
PyGpuNdArray_Zeros(int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran);

static PyObject *
PyGpuNdArray_Empty(int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran);

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
