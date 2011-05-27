#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>
#include <iostream>

#include "pygpu_ndarray.cuh"
#include "pygpu_language.h"

//#include "pygpu_ndarray_ctor.cu"//TODO correctly handle the compilation...


/////////////////////////
// Static helper methods
/////////////////////////

static void
PyGpuNdArray_null_init(PyGpuNdArrayObject *self)
{
    if(0) fprintf(stderr, "PyGpuNdArrayObject_null_init\n");

    PyGpuNdArray_DATA(self) = NULL;
    PyGpuNdArray_OFFSET(self) = 0;
    PyGpuNdArray_NDIM(self) = -1;
    self->base = NULL;
    PyGpuNdArray_DIMS(self) = NULL;
    PyGpuNdArray_STRIDES(self) = NULL;
    PyGpuNdArray_FLAGS(self) = NPY_DEFAULT;
    self->descr = NULL;

    self->data_allocated = 0;
}



/////////////////////////////
// Satisfying reqs to be Type
/////////////////////////////

//DON'T use directly(if their is other PyGpuNdArrayObject that point to it, it will cause problem)! use Py_DECREF() instead
static void
PyGpuNdArrayObject_dealloc(PyGpuNdArrayObject* self)
{
    if(0) fprintf(stderr, "PyGpuNdArrayObject_dealloc\n");
    if (0) std::cerr << "PyGpuNdArrayObject dealloc " << self << " "<<self->data_allocated<<'\n';
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
    //Py_XDECREF(self->descr);//TODO: How to handle the refcont on this object?
    self->descr = NULL;
    self->data_allocated = 0;

    self->ob_type->tp_free((PyObject*)self);
    --_outstanding_mallocs[1];
    if(0){
        fprintf(stderr, "device_malloc_counts: (device) %i (obj) %i\n",
                _outstanding_mallocs[0],
                _outstanding_mallocs[1]);
    }
    if(0) fprintf(stderr, "PyGpuNdArrayObject_dealloc end\n");
}

static PyObject *
PyGpuNdArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    if(0) fprintf(stderr, "PyGpuNdArray_new\n");
    PyGpuNdArrayObject *self;

    self = (PyGpuNdArrayObject *)type->tp_alloc(type, 0);
    if (self != NULL){
        PyGpuNdArray_null_init(self);
        ++_outstanding_mallocs[1];
    }
    if(0) fprintf(stderr, "PyGpuNdArray_new end %p\n", self);
    return (PyObject *)self;
}

static int
PyGpuNdArray_init(PyGpuNdArrayObject *self, PyObject *args, PyObject *kwds)
{
    if(0) fprintf(stderr, "PyGpuNdArray_init\n");
    PyObject *arr=NULL;

    if (! PyArg_ParseTuple(args, "O", &arr))
        return -1;
    if (! PyArray_Check(arr)){
        PyErr_SetString(PyExc_TypeError, "PyGpuNdArrayObject_init: PyArray or PyGpuNdArrayObject arg required");
        return -1;
    }

    // TODO: We must create a new copy of the PyArray_Descr(or this only increment the refcount?) or still the reference?
    PyArray_Descr * type = PyArray_DescrFromType(PyArray_TYPE(arr));
    self->descr = type;
    Py_XINCREF(self->descr);//TODO: How to handle the refcont on this object?
    int rval = PyGpuNdArray_CopyFromArray(self, (PyArrayObject*)arr);
    if(0) fprintf(stderr, "PyGpuNdArray_init: end %p type=%p\n", self, self->descr);
    return rval;
}


int
PyGpuNdArray_CopyFromArray(PyGpuNdArrayObject * self, PyArrayObject*obj)
{
    int verbose = 0;
    if (verbose) fprintf(stderr, "PyGpuNdArray_CopyFromArray: start descr=%p\n", self->descr);
    //modif done to the new array won't be updated!
    assert(!PyGpuNdArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    //Aligned are not tested, so don't allow it for now
    assert(PyGpuNdArray_CHKFLAGS(self, NPY_ALIGNED));

    int typenum = PyArray_TYPE(obj);
    PyObject * py_src = NULL;
    if (PyArray_ISONESEGMENT(obj)) {
        Py_INCREF(obj);
        py_src = (PyObject *) obj;
    }else{
        py_src = PyArray_ContiguousFromAny((PyObject*)obj, typenum, PyArray_NDIM(obj), PyArray_NDIM(obj));
    }
    if (verbose) fprintf(stderr, "PyGpuNdArray_CopyFromArray: contiguous!\n");
    if (!py_src) {
        return -1;
    }

    int err;
    if(PyArray_ISFORTRAN(obj) && ! PyArray_ISCONTIGUOUS(obj)){
        if (verbose) fprintf(stderr, "PyGpuNdArray_CopyFromArray: fortran!\n");
        err = PyGpuNdArray_alloc_contiguous(self, obj->nd, obj->dimensions, NPY_FORTRANORDER);
    }else{
        err = PyGpuNdArray_alloc_contiguous(self, obj->nd, obj->dimensions);
    }
    if (err) {
        return err;
    }

    //check that the flag are the same
    if (PyArray_ISCONTIGUOUS(py_src) != PyGpuNdArray_ISCONTIGUOUS(self) &&
        PyArray_ISFORTRAN(obj) && 0) {
        PyErr_Format(PyExc_RuntimeError, "ISCONTIGUOUS %d %d\n", PyArray_ISCONTIGUOUS(py_src), PyGpuNdArray_ISCONTIGUOUS(self));
        return -1;
    }
    assert(PyArray_ISCONTIGUOUS(py_src) == PyGpuNdArray_ISCONTIGUOUS(self) ||
           PyArray_ISFORTRAN(obj));
    assert(PyArray_ISFORTRAN(py_src) == PyGpuNdArray_ISFORTRAN(self));
    assert(PyArray_ISALIGNED(py_src) == PyGpuNdArray_ISALIGNED(self));

    // New memory, so we should own it.
    assert(PyGpuNdArray_CHKFLAGS(self, NPY_OWNDATA));
    // New memory, so it should be writable
    assert(PyGpuNdArray_ISWRITEABLE(self));


    cublasSetVector(PyArray_SIZE(py_src),
		    PyArray_ITEMSIZE(py_src),
		    PyArray_DATA(py_src), 1,
		    PyGpuNdArray_DATA(self), 1);
    CNDA_THREAD_SYNC;
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to device memory");
        Py_DECREF(py_src);
        return -1;
    }
    Py_DECREF(py_src);
    if (verbose) fprintf(stderr, "PyGpuNdArray_CopyFromArray: end\n");
    return 0;
}

static PyObject * PyGpuNdArray_Copy(PyGpuNdArrayObject * self)
{
    int verbose = 0;
    if (verbose) fprintf(stderr, "PyGpuNdArray_Copy start\n");
    PyObject * rval = PyGpuNdArray_New();
    //TODO find how to refcount descr.
    PyGpuNdArray_DESCR(rval) = PyGpuNdArray_DESCR(self);
    if ((!rval) || (-1 == PyGpuNdArray_NDIM(self))) {
        return rval;
    }
    if (PyGpuNdArray_alloc_contiguous((PyGpuNdArrayObject*)rval, PyGpuNdArray_NDIM(self), PyGpuNdArray_DIMS(self))) {
        Py_DECREF(rval);
        return NULL;
    }

    if (PyGpuNdArray_CopyFromPyGpuNdArray((PyGpuNdArrayObject*)rval, self)) {
        Py_DECREF(rval);
        return NULL;
    }
    if (verbose>1) PyGpuNdArray_fprint(stderr, self);
    if (verbose>1) PyGpuNdArray_fprint(stderr, (PyGpuNdArrayObject *)rval);
    if (verbose) fprintf(stderr, "PyGpuNdArray_Copy end\n");
    return rval;
}

//updated for offset
PyObject * PyGpuNdArray_CreateArrayObj(PyGpuNdArrayObject * self)
{
    int verbose = 0;

    if(verbose) fprintf(stderr, "PyGpuNdArray_CreateArrayObj\n");

    assert(PyGpuNdArray_OFFSET(self)==0);//TODO implement when offset is not 0!

    if(PyGpuNdArray_NDIM(self)>=0 && PyGpuNdArray_SIZE(self)==0){
      npy_intp * npydims = (npy_intp*)malloc(PyGpuNdArray_NDIM(self) * sizeof(npy_intp));
      assert (npydims);
      for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
	npydims[i] = (npy_intp)(PyGpuNdArray_DIMS(self)[i]);

      // Numpy will do a decref on the description.
      Py_INCREF(PyGpuNdArray_DESCR(self));
      PyObject * rval = PyArray_Empty(PyGpuNdArray_NDIM(self),
				      npydims, self->descr,
				      PyGpuNdArray_ISFARRAY(self));
      free(npydims);
      if (!rval){
        return NULL;
      }
      assert (PyArray_ITEMSIZE(rval) == PyGpuNdArray_ITEMSIZE(self));
      return rval;
    }
    if ((PyGpuNdArray_NDIM(self) < 0) || (PyGpuNdArray_DATA(self) == 0)) {
        PyErr_SetString(PyExc_ValueError, "can't copy from un-initialized PyGpuNdArray");
        return NULL;
    }
    PyGpuNdArrayObject * contiguous_self = NULL;
    bool pos_stride = true;
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
        if (PyGpuNdArray_STRIDE(self,i)<0)
            pos_stride = false;
    if (PyGpuNdArray_ISONESEGMENT(self) && pos_stride) {
        contiguous_self = self;
        Py_INCREF(contiguous_self);
        if (verbose) std::cerr << "PyGpuNdArray_CreateArrayObj: gpu array already contiguous" <<
		       contiguous_self << '\n';
        //}else if(PyGpuNdArray_ISONESEGMENT(self)){
        //TODO implement special object handling to speed up transfer
        //  if (verbose) std::cerr << "CreateArrayObj one segment, with special handling" << contiguous_self << '\n';
        //PyErr_SetString(PyExc_ValueError, "PyGpuNdArray_CreateArrayObj: Need PyGpuNdArray_Copy or some other nd array mandling to transfer contiguous bloc with negative stride.");
        //return NULL;
    } else {
        contiguous_self = (PyGpuNdArrayObject*)PyGpuNdArray_Copy(self);
        if (verbose) std::cerr << "CreateArrayObj created contiguous" << contiguous_self << '\n';
    }
    if (!contiguous_self) {
        return NULL;
    }

    npy_intp * npydims = (npy_intp*)malloc(PyGpuNdArray_NDIM(self) * sizeof(npy_intp));
    assert (npydims);
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i) npydims[i] = (npy_intp)(PyGpuNdArray_DIMS(self)[i]);
    Py_INCREF(PyGpuNdArray_DESCR(self));
    PyObject * rval = PyArray_Empty(PyGpuNdArray_NDIM(self),
				    npydims,
				    PyGpuNdArray_DESCR(self),
				    PyGpuNdArray_ISFORTRAN(self));
    free(npydims);
    if (!rval) {
        Py_DECREF(contiguous_self);
        return NULL;
    }

    cublasGetVector(PyArray_SIZE(rval), PyArray_ITEMSIZE(rval),
		    PyGpuNdArray_DATA(contiguous_self), 1,
		    PyArray_DATA(rval), 1);
    CNDA_THREAD_SYNC;

    if (CUBLAS_STATUS_SUCCESS != cublasGetError()) {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to host");
        Py_DECREF(contiguous_self);
        Py_DECREF(rval);
        rval = NULL;
    }

    Py_DECREF(contiguous_self);
    return rval;
}

static PyMethodDef PyGpuNdArray_methods[] =
{
    {"__array__",
        (PyCFunction)PyGpuNdArray_CreateArrayObj, METH_NOARGS,
        "Copy from the device to a numpy ndarray"},
    {"copy",
        (PyCFunction)PyGpuNdArray_Copy, METH_NOARGS,
        "Create a deep copy of this object."},
/*
    {"__copy__",
        (PyCFunction)PyGpuNdArray_View, METH_NOARGS,
        "Create a shallow copy of this object. used by module copy"},
    {"__deepcopy__",
        (PyCFunction)PyGpuNdArray_DeepCopy, METH_O,
        "Create a copy of this object"},
    {"zeros",
        (PyCFunction)PyGpuNdArray_Zeros, METH_STATIC,
        "Create a new PyGpuNdArray with specified shape, filled with zeros."},
    {"copy",
        (PyCFunction)PyGpuNdArray_Copy, METH_NOARGS,
        "Create a copy of this object"},
    {"reduce_sum",
        (PyCFunction)PyGpuNdArray_ReduceSum, METH_O,
        "Reduce over the given dimensions by summation"},
    {"exp",
        (PyCFunction)PyGpuNdArray_Exp, METH_NOARGS,
        "Return the exponential of all elements"},
    {"reshape",
        (PyCFunction)PyGpuNdArray_Reshape, METH_O,
        "Return a reshaped view (or copy) of this ndarray\n\
            The required argument is a tuple of integers specifying the shape of the new ndarray."},
    {"view",
        (PyCFunction)PyGpuNdArray_View, METH_NOARGS,
        "Return an alias of this ndarray"},
    {"_set_stride",
        (PyCFunction)PyGpuNdArray_SetStride, METH_VARARGS,
        "For integer arguments (i, s), set the 'i'th stride to 's'"},
    {"_set_shape_i",
        (PyCFunction)PyGpuNdArray_SetShapeI, METH_VARARGS,
        "For integer arguments (i, s), set the 'i'th shape to 's'"},
    */
    {NULL, NULL, NULL, NULL}  /* Sentinel */
};

//PyArray_CopyInto(PyArrayObject* dest, PyArrayObject* src)¶
//PyObject* PyArray_NewCopy(PyArrayObject* old, NPY_ORDER order)¶


static PyObject *
PyGpuNdArray_get_shape(PyGpuNdArrayObject *self, void *closure)
{
    if(0) fprintf(stderr, "PyGpuNdArray_get_shape\n");

    if (PyGpuNdArray_NDIM(self) < 0)
    {
        PyErr_SetString(PyExc_ValueError, "PyGpuNdArray not initialized");
        return NULL;
    }
    PyObject * rval = PyTuple_New(PyGpuNdArray_NDIM(self));
    for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
    {
        if (!rval || PyTuple_SetItem(rval, i, PyInt_FromLong(PyGpuNdArray_DIMS(self)[i])))
        {
            Py_XDECREF(rval);
            return NULL;
        }

    }
    return rval;
}

static int
PyGpuNdArray_set_shape(PyGpuNdArrayObject *self, PyObject *value, void *closure)
{
    PyErr_SetString(PyExc_NotImplementedError, "TODO: call reshape");
    return -1;
}

static PyObject *
PyGpuNdArray_get_strides(PyGpuNdArrayObject *self, void *closure)
{
  if ( PyGpuNdArray_NDIM(self) < 0){
      PyErr_SetString(PyExc_ValueError, "PyGpuNdArrayObject not initialized");
      return NULL;
    }
  PyObject * rval = PyTuple_New( PyGpuNdArray_NDIM(self));
  for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i){
      if (!rval || PyTuple_SetItem(rval, i, PyInt_FromLong(PyGpuNdArray_STRIDES(self)[i]))){
	  Py_XDECREF(rval);
	  return NULL;
        }
    }
  return rval;
}

static PyObject *
PyGpuNdArray_get_data(PyGpuNdArrayObject *self, void *closure)
{
    return PyInt_FromLong((long int) PyGpuNdArray_DATA(self));
}

static PyObject *
PyGpuNdArray_get_flags(PyGpuNdArrayObject *self, void *closure)
{
    PyObject * dict = PyDict_New();

    PyObject * str= PyString_FromString("C_CONTIGUOUS");
    PyObject * i = PyBool_FromLong(PyGpuNdArray_ISCONTIGUOUS(self));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    str= PyString_FromString("F_CONTIGUOUS");
    i = PyBool_FromLong(PyGpuNdArray_CHKFLAGS(self, NPY_F_CONTIGUOUS));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    str= PyString_FromString("WRITEABLE");
    i = PyBool_FromLong(PyGpuNdArray_ISWRITEABLE(self));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    str= PyString_FromString("ALIGNED");
    i = PyBool_FromLong(PyGpuNdArray_ISALIGNED(self));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    str= PyString_FromString("UPDATEIFCOPY");
    i = PyBool_FromLong(PyGpuNdArray_CHKFLAGS(self, NPY_UPDATEIFCOPY));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    str= PyString_FromString("OWNDATA");
    i = PyBool_FromLong(PyGpuNdArray_CHKFLAGS(self, NPY_OWNDATA));
    PyDict_SetItem(dict, str, i);
    Py_DECREF(str);
    Py_DECREF(i);

    return dict;
}
static PyObject *
PyGpuNdArray_get_ndim(PyGpuNdArrayObject *self, void *closure)
{
    return PyInt_FromLong((long int) PyGpuNdArray_NDIM(self));
}
static PyObject *
PyGpuNdArray_get_offset(PyGpuNdArrayObject *self, void *closure)
{
    return PyInt_FromLong((long int) PyGpuNdArray_OFFSET(self));
}
static PyObject *
PyGpuNdArray_get_data_allocated(PyGpuNdArrayObject *self, void *closure)
{
    return PyInt_FromLong((long int) self->data_allocated);
}
static PyObject *
PyGpuNdArray_get_size(PyGpuNdArrayObject *self, void *closure)
{
    return PyInt_FromLong((long int) PyGpuNdArray_SIZE(self));
}

static PyObject *
PyGpuNdArray_get_base(PyGpuNdArrayObject *self, void *closure)
{
    if (!PyGpuNdArray_BASE(self)){
        Py_INCREF(Py_None);
        return Py_None;
    }
    PyObject * ret = PyGpuNdArray_BASE(self);
    Py_INCREF(ret);
    return ret;
}

static PyObject *
PyGpuNdArray_get_dtype(PyArrayObject *self)
{
    Py_INCREF(PyGpuNdArray_DESCR(self));
    PyObject * ret = (PyObject *)PyGpuNdArray_DESCR(self);
    return ret;
}

static PyObject *
PyGpuNdArray_get_itemsize(PyArrayObject *self)
{
    return (PyObject *)PyGpuNdArray_ITEMSIZE(self);
}

static PyGetSetDef PyGpuNdArray_getset[] = {
    {"base",
        (getter)PyGpuNdArray_get_base,
        NULL,
        "Return the object stored in the base attribute",
        NULL},
    {"bytes",
        (getter)PyGpuNdArray_get_data,
        NULL,
        "device data pointer",
        NULL},
    {"shape",
        (getter)PyGpuNdArray_get_shape,
        (setter)PyGpuNdArray_set_shape,
        "shape of this ndarray (tuple)",
        NULL},
    {"strides",
        (getter)PyGpuNdArray_get_strides,
        NULL,//(setter)PyGpuNdArray_set_strides,
        "data pointer strides (in elements)",
        NULL},
    {"ndim",
        (getter)PyGpuNdArray_get_ndim,
        NULL,
        "The number of dimensions in this object",
        NULL},
    {"offset",
        (getter)PyGpuNdArray_get_offset,
        NULL,
        "Return the offset value",
        NULL},
    {"size",
        (getter)PyGpuNdArray_get_size,
        NULL,
        "The number of elements in this object.",
        NULL},
    {"data_allocated",
        (getter)PyGpuNdArray_get_data_allocated,
        NULL,
        "The size of the allocated memory on the device.",
        NULL},
    {"itemsize",
        (getter)PyGpuNdArray_get_itemsize,
        NULL,
        "The size of the base element.",
        NULL},
    {"dtype",
	(getter)PyGpuNdArray_get_dtype,
	NULL,
	"The dtype of the element",
	NULL},
    {"flags",
        (getter)PyGpuNdArray_get_flags,
        NULL,
        "Return the flags as a dictionary",
        NULL},
    {NULL, NULL, NULL, NULL}  /* Sentinel */
};

// Will by called by __len__ in Python
static Py_ssize_t
PyGpuNdArray_len(PyObject * py_self)
{
    PyGpuNdArrayObject * self = (PyGpuNdArrayObject*) py_self;
    if (PyGpuNdArray_NDIM(self) <= 0)
    {
        return (Py_ssize_t) 0;
    }
    else
    {
        return (Py_ssize_t) PyGpuNdArray_DIMS(self)[0];
    }
}

static int
PyGpuNdArray_set_data(PyGpuNdArrayObject * self, char * data, PyObject * base)
{
    if (self->data_allocated)
    {
        assert(PyGpuNdArray_DATA(self));
        if (device_free(PyGpuNdArray_DATA(self)))
        {
            PyGpuNdArray_DATA(self) = NULL;
            self->data_allocated = 0;
            return -1;
        }
    }
    //N.B. XDECREF and XINCREF are no-ops for NULL pointers
    if (PyGpuNdArray_BASE(self) != base)
    {
        Py_XDECREF(PyGpuNdArray_BASE(self));
        PyGpuNdArray_BASE(self) = base;
        Py_XINCREF(PyGpuNdArray_BASE(self));
    }
    self->data_allocated = 0;
    PyGpuNdArray_DATA(self) = data;
    return 0;
}

// Will by called by __getitem__ in Python
static PyObject *
PyGpuNdArray_Subscript(PyObject * py_self, PyObject * key)
{
    int verbose = 0;
    if (verbose) fprintf(stderr, "Subscript .... \n");
    PyGpuNdArrayObject * self = (PyGpuNdArrayObject*) py_self;
    PyObject * py_rval = NULL;
    PyGpuNdArrayObject * rval = NULL;
    PyObject * intobj = NULL;

    //PyObject_Print(key, stderr, 0);

    assert(PyGpuNdArray_OFFSET(self)==0);

    if (key == Py_Ellipsis)
    {
        if (verbose) fprintf(stderr, "Subscript with ellipse \n");
        Py_INCREF(py_self);
        if (verbose) fprintf(stderr, "Subscript with ellipse end\n");
        return py_self;
    }
    if ((intobj=PyNumber_Int(key))) //INDEXING BY INTEGER
    {
        if (verbose>1) PyGpuNdArray_fprint(stderr, self);
        if (verbose) fprintf(stderr, "Subscript with int \n");

        int d_idx = PyInt_AsLong(intobj);
        Py_DECREF(intobj); intobj=NULL;

        if (verbose) fprintf(stderr, "Subscript with int 1\n");
        if (PyGpuNdArray_NDIM(self) == 0) {
            PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
            return NULL;
        }else if (PyGpuNdArray_NDIM(self)< 0){
            PyErr_SetString(PyExc_IndexError, "nd arrays must have a number of dim > 0!");
            return NULL;
        }
        int d_dim = PyGpuNdArray_DIMS(self)[0];
        int offset = 0;
        if (verbose) fprintf(stderr, "Subscript with int 2\n");

        if ((d_idx >= 0) && (d_idx < d_dim)) {
            //normal indexing
            offset += d_idx * PyGpuNdArray_STRIDES(self)[0];
        }
        else if ((d_idx < 0) && (d_idx >= -d_dim)) {
            //end-based indexing
            // d_idx is negative
            offset += (d_dim + d_idx) * PyGpuNdArray_STRIDES(self)[0];
        } else {
            PyErr_SetString(PyExc_IndexError, "index out of bounds");
            return NULL;
        }
        if (verbose) fprintf(stderr, "Subscript with int 3\n");

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(PyGpuNdArray_NDIM(self) - 1);
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);

        if (verbose) fprintf(stderr, "Subscript with int 4\n");
        //initialize the view's data pointer to our own.
        assert (0 == rval->data_allocated);
        if (PyGpuNdArray_set_data(rval, PyGpuNdArray_DATA(self) + offset, (PyObject *) self)){
            Py_DECREF(rval);
            return NULL;
        }
        if (verbose) fprintf(stderr, "Subscript with int 5\n");

        for (int d = 1; d < PyGpuNdArray_NDIM(self); ++d) {
            PyGpuNdArray_STRIDE(rval, d-1) = PyGpuNdArray_STRIDES(self)[d];
            PyGpuNdArray_DIM(rval, d-1) = PyGpuNdArray_DIMS(self)[d];
        }
    }
    else {
        PyErr_Clear();
    }
    if (PySlice_Check(key)) //INDEXING BY SLICE
    {
        if (verbose) fprintf(stderr, "Subscript with slice \n");
        if (PyGpuNdArray_NDIM(self) == 0)
        {
            PyErr_SetString(PyExc_ValueError, "cannot slice a 0-d array");
            return NULL;
        }

        int d_dim = PyGpuNdArray_DIMS(self)[0];
        Py_ssize_t start, stop, step, slen;
        if (PySlice_GetIndicesEx((PySliceObject*)key, d_dim, &start, &stop, &step, &slen)) {
            return NULL;
        }
        if (verbose>2) {
            std::cerr << "start " << start << "\n";
            std::cerr << "stop " << stop << "\n";
            std::cerr << "step " << step << "\n";
            std::cerr << "slen " << slen << "\n";
        }

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(PyGpuNdArray_NDIM(self));
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);
        assert (0 == rval->data_allocated);
        if (PyGpuNdArray_set_data(rval,
                                  PyGpuNdArray_DATA(self) + start * PyGpuNdArray_STRIDE(self, 0),
                                  py_self)) {
            Py_DECREF(rval);
            return NULL;
        }

        //initialize dimension 0 of rval
        PyGpuNdArray_STRIDE(rval, 0) = step * PyGpuNdArray_STRIDES(self)[0];
        PyGpuNdArray_DIM(rval, 0) = slen;
        if (verbose) std::cerr << "rval stride " << PyGpuNdArray_STRIDES(rval)[0] << "\n";
        // initialize dimensions > 0 of rval
        for (int d = 1; d < PyGpuNdArray_NDIM(self); ++d) {
            PyGpuNdArray_STRIDE(rval, d) = PyGpuNdArray_STRIDES(self)[d];
            PyGpuNdArray_DIM(rval, d) = PyGpuNdArray_DIMS(self)[d];
        }
    }
    if (PyTuple_Check(key)) //INDEXING BY TUPLE
    {
        if (verbose) fprintf(stderr, "Subscript with tuple \n");
        //elements of the tuple can be either integers or slices
        //the dimensionality of the view we will return is diminished for each slice in the tuple

        if (PyTuple_Size(key) > PyGpuNdArray_NDIM(self))
        {
            PyErr_SetString(PyExc_IndexError, "index error");
            return NULL;
        }

        //calculate the number of dimensions in the return value
        int rval_nd = PyGpuNdArray_NDIM(self);
        for (int d = 0; d < PyTuple_Size(key); ++d)
        {
            //On some paltform PyInt_Check(<type 'numpy.int64'>) return true, other it return false.
            //So we use PyArray_IsAnyScalar that should covert everything.
            rval_nd -= PyArray_IsAnyScalar(PyTuple_GetItem(key, d));
        }

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(rval_nd);
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;
        assert (0 == rval->data_allocated);

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);

        //initialize the view's data pointer to our own.
        if (PyGpuNdArray_set_data(rval, PyGpuNdArray_DATA(self), py_self))
        {
            Py_DECREF(rval);
            return NULL;
        }

        // rval_d will refer to the current dimension in the rval.
        // It will not be incremented for integer keys, but will be incremented for slice
        // keys
        int rval_d = 0;

        for (int d = 0; d < PyGpuNdArray_NDIM(self); ++d)
        {
            // keys can be shorter than PyGpuNdArray_NDIM(self).
            // when that happens, it means that the remaining dimensions are "full slices"
            if (d >=PyTuple_Size(key))
            {
                PyGpuNdArray_STRIDE(rval, rval_d) = PyGpuNdArray_STRIDES(self)[d];
                PyGpuNdArray_DIM(rval, rval_d) = PyGpuNdArray_DIMS(self)[d];
                ++rval_d;
            }
            else
            {
                PyObject * key_d = PyTuple_GetItem(key, d);

                if (PySlice_Check(key_d))
                {
                    Py_ssize_t start, stop, step, slen;
                    if (PySlice_GetIndicesEx((PySliceObject*)key_d, PyGpuNdArray_DIMS(self)[d], &start, &stop, &step, &slen))
                    {
                        Py_DECREF(rval);
                        return NULL;
                    }
                    PyGpuNdArray_DATA(rval) += start * PyGpuNdArray_STRIDES(self)[d];
                    PyGpuNdArray_STRIDE(rval, rval_d) = step * PyGpuNdArray_STRIDES(self)[d];
                    PyGpuNdArray_DIM(rval, rval_d) = slen;
                    if (0)
                    {
                        std::cerr << "start " << start << "\n";
                        std::cerr << "stop " << stop << "\n";
                        std::cerr << "step " << step << "\n";
                        std::cerr << "slen " << slen << "\n";
                    }
                    ++rval_d;
                }
                else if ((intobj=PyNumber_Int(key_d)))
                {
                    assert(PyArray_IsAnyScalar(key_d));
                    int d_idx = PyInt_AsLong(intobj);
                    Py_DECREF(intobj);
                    intobj = NULL;
                    int d_dim = PyGpuNdArray_DIMS(self)[d];

                    if ((d_idx >= 0) && (d_idx < d_dim))
                    {
                        //normal indexing
                        PyGpuNdArray_DATA(rval) += d_idx * PyGpuNdArray_STRIDES(self)[d];
                    }
                    else if ((d_idx < 0) && (d_idx >= -d_dim))
                    {
                        //end-based indexing
                        PyGpuNdArray_DATA(rval) += (d_dim + d_idx) * PyGpuNdArray_STRIDES(self)[d];
                    }
                    else
                    {
                        PyErr_SetString(PyExc_IndexError, "index out of bounds");
                        Py_DECREF(rval);
                        return NULL;
                    }
                }
                else
                {
                    PyErr_Clear(); // clear the error set by PyNumber_Int
                    PyErr_SetString(PyExc_IndexError, "index must be either int or slice");
                    Py_DECREF(rval);
                    return NULL;
                }
            }
        }
    }
    if (py_rval)
    {
        if (verbose>1) PyGpuNdArray_fprint(stderr, self);
        if (verbose>1) PyGpuNdArray_fprint(stderr, rval);
    }
    else
    {
        PyErr_SetString(PyExc_NotImplementedError, "Unknown key type");
        return NULL;
    }

    // Set flags
    if (PyGpuNdArray_ISWRITEABLE(self)) {
        PyGpuNdArray_FLAGS(rval) |= NPY_WRITEABLE;
    } else {
        PyGpuNdArray_FLAGS(rval) &= ~NPY_WRITEABLE;
    }
    PyGpuNdArray_FLAGS(rval) &= ~NPY_OWNDATA;
    if (PyGpuNdArray_ISALIGNED(self)) {
        PyGpuNdArray_FLAGS(rval) |= NPY_ALIGNED;
    } else {
        PyGpuNdArray_FLAGS(rval) &= ~NPY_ALIGNED;
    }
    PyGpuNdArray_FLAGS(rval) &= ~NPY_UPDATEIFCOPY;

    if (false && PyGpuNdArray_NDIM(rval) == 0) {
        //Numpy is not consistent here
        //When we create a new numpy ndarray of 0 dim, it is not f contiguous
        //But when we take a subtensor that is of 0 dim, it is f contiguous!
        //We make as them for now...
        PyGpuNdArray_FLAGS(rval) &= ~NPY_F_CONTIGUOUS;
        PyGpuNdArray_FLAGS(rval) |= NPY_C_CONTIGUOUS;
    } else {
        if (PyGpuNdArray_is_c_contiguous(rval)) {
            PyGpuNdArray_FLAGS(rval) |= NPY_C_CONTIGUOUS;
        } else {
            PyGpuNdArray_FLAGS(rval) &= ~NPY_C_CONTIGUOUS;
        }
        if (PyGpuNdArray_is_f_contiguous(rval)) {
            PyGpuNdArray_FLAGS(rval) |= NPY_F_CONTIGUOUS;
        } else {
            PyGpuNdArray_FLAGS(rval) &= ~NPY_F_CONTIGUOUS;
        }
    }

    if (verbose) fprintf(stderr, "Subscript end\n");
    return py_rval;
}

PyMappingMethods PyGpuNdArrayMappingMethods = {
    PyGpuNdArray_len, //lenfunc mp_length;               __len__
    PyGpuNdArray_Subscript, //binaryfunc mp_subscript;   __getitem__
    0 //PyGpuNdArray_setitem //objobjargproc mp_ass_subscript;                __setitem__
};

static PyTypeObject PyGpuNdArrayType =
{
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "GpuNdArray",             /*tp_name*/
    sizeof(PyGpuNdArrayObject),       /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyGpuNdArrayObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0, //&PyGpuNdArrayObjectNumberMethods, /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    &PyGpuNdArrayMappingMethods,/*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
    "PyGpuNdArrayObject objects",     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyGpuNdArray_methods,       /* tp_methods */
    0, //PyGpuNdArray_members,       /* tp_members */ //TODO
    PyGpuNdArray_getset,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyGpuNdArray_init,/* tp_init */
    0,                         /* tp_alloc */
    PyGpuNdArray_new,           /* tp_new */
};

//////////////////////////////////////
//
// C API FOR PyGpuNdArrayObject
//
//////////////////////////////////////
PyObject *
PyGpuNdArray_New(int nd)
{
    PyGpuNdArrayObject *self = (PyGpuNdArrayObject *)PyGpuNdArrayType.tp_alloc(&PyGpuNdArrayType, 0);
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "PyGpuNdArray_New failed to allocate self");
        return NULL;
    }
    PyGpuNdArray_null_init(self);

    if (nd == 0) {
        PyGpuNdArray_NDIM(self) = 0;
    }
    else if (nd > 0) {
        if (PyGpuNdArray_set_nd(self, nd)) {
            Py_DECREF(self);
            return NULL;
        }
    }
    ++_outstanding_mallocs[1];
    return (PyObject *)self;
}

int
PyGpuNdArray_Check(const PyObject * ob)
{
    if(0) fprintf(stderr, "PyGpuNdArray_Check\n");
    //TODO: doesn't work with inheritance
    return PyGpuNdArray_CheckExact(ob);
}
int
PyGpuNdArray_CheckExact(const PyObject * ob)
{
    if(0) fprintf(stderr, "PyGpuNdArray_CheckExact\n");
    return ((ob->ob_type == &PyGpuNdArrayType) ? 1 : 0);
}


static PyMethodDef module_methods[] = {
    //{"dimshuffle", PyGpuNdArray_Dimshuffle, METH_VARARGS, "Returns the dimshuffle of a PyGpuNdArray."},
    {"outstanding_mallocs", outstanding_mallocs, METH_VARARGS, "how many more mallocs have been called than free's"},
    {NULL, NULL, NULL, NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initpygpu_ndarray(void)
{
    import_array();

    PyObject* m;

    if (PyType_Ready(&PyGpuNdArrayType) < 0)
        return;

    m = Py_InitModule3("pygpu_ndarray", module_methods,
                       "Example module that creates an extension type.");

    if (m == NULL)
        return;

    Py_INCREF(&PyGpuNdArrayType);
    PyModule_AddObject(m, "GpuNdArrayObject", (PyObject *)&PyGpuNdArrayType);
#if COMPUTE_GPU_MEM_USED
    for(int i=0;i<TABLE_SIZE;i++){
      _alloc_size_table[i].ptr=NULL;
      _alloc_size_table[i].size=0;
    }
#endif
    //    cublasInit();
    //if (0&&CUBLAS_STATUS_SUCCESS != cublasGetError())
    //{
        //std::cerr << "WARNING: initcuda_ndarray: error initializing device\n";
    //}
/*
    if (0) //TODO: is this necessary?
    {
        int deviceId = 0; // TODO: what number goes here?
        cudaSetDevice(deviceId);
        cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err)
        {
            std::cerr << "Error in SetDevice:" << cudaGetErrorString(err) << "\n";
        }
    }
*/
}

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
