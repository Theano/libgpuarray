#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>
#include <iostream>

#include "pygpu_ndarray.h"
#include "pygpu_language.h"

/////////////////////////
// Static helper methods
/////////////////////////

static void
PyGpuNdArray_null_init(PyGpuNdArrayObject *self)
{
    DPRINTF("PyGpuNdArrayObject_null_init\n");

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
    DPRINTF("PyGpuNdArrayObject_dealloc\n");
    DPRINTF("PyGpuNdArrayObject dealloc %p %d %p\n", self, self->data_allocated, PyGpuNdArray_DATA(self));

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
    DPRINTF("device_malloc_counts: (device) %i (obj) %i\n",
            _outstanding_mallocs[0],
            _outstanding_mallocs[1]);
    DPRINTF("PyGpuNdArrayObject_dealloc end\n");
}

static PyObject *
PyGpuNdArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    DPRINTF("PyGpuNdArray_new\n");
    PyGpuNdArrayObject *self;

    self = (PyGpuNdArrayObject *)type->tp_alloc(type, 0);
    if (self != NULL){
        PyGpuNdArray_null_init(self);
        ++_outstanding_mallocs[1];
    }
    DPRINTF("PyGpuNdArray_new end %p\n", self);
    return (PyObject *)self;
}

static int
PyGpuNdArray_init(PyGpuNdArrayObject *self, PyObject *args, PyObject *kwds)
{
    DPRINTF("PyGpuNdArray_init\n");
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
    DPRINTF("PyGpuNdArray_init: end %p type=%p\n", self, self->descr);
    return rval;
}


int
PyGpuNdArray_CopyFromArray(PyGpuNdArrayObject * self, PyArrayObject*obj)
{
    DPRINTF("PyGpuNdArray_CopyFromArray: start descr=%p\n", self->descr);
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
        py_src = PyArray_ContiguousFromAny((PyObject*)obj, typenum,
                                           PyArray_NDIM(obj),
                                           PyArray_NDIM(obj));
    }
    DPRINTF("PyGpuNdArray_CopyFromArray: contiguous!\n");
    if (!py_src) {
        return -1;
    }

    int err;
    if(PyArray_ISFORTRAN(obj) && ! PyArray_ISCONTIGUOUS(obj)){
        DPRINTF("PyGpuNdArray_CopyFromArray: fortran!\n");
        err = PyGpuNdArray_alloc_contiguous(self, obj->nd, obj->dimensions,
                                            NPY_FORTRANORDER);
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

    err = PyGpuMemcpy(PyGpuNdArray_DATA(self),
                      PyArray_DATA(py_src),
                      PyGpuNdArray_OFFSET(self),
                      PyArray_SIZE(py_src) * PyArray_ITEMSIZE(py_src),
                      PyGpuHostToDevice);
    if (err) {
        Py_DECREF(py_src);
        return -1;
    }
    Py_DECREF(py_src);
    DPRINTF("PyGpuNdArray_CopyFromArray: end\n");
    return 0;
}

static PyObject * PyGpuNdArray_copy(PyObject * self, PyObject *args,
                                    PyObject *kargs)
{
    DPRINTF("PyGpuNdArray_copy start\n");
    static const char *kwlist[] = {"order", NULL};
    NPY_ORDER order = PyArray_CORDER;

    if(!PyGpuNdArray_Check(self)){
        PyErr_SetString(PyExc_ValueError, "PyGpuNdArray_copy: expected a PyGpuNdArrayObject.");
        return NULL;
    }

    DPRINTF("PyGpuNdArray_copy before parse inputs\n");
    if (!PyArg_ParseTupleAndKeywords(args, kargs, "|O&",
                                     (char**)kwlist,
                                     PyArray_OrderConverter,
                                     &order)) {
        DPRINTF("PyGpuNdArray_copy start1.2\n");
        return NULL;
    }
    DPRINTF("PyGpuNdArray_copy after parse inputs\n");

    DPRINTF("PyGpuNdArray_copy before copy\n");
    PyObject *ret = PyGpuNdArray_Copy((PyGpuNdArrayObject*)self, order);
    DPRINTF("PyGpuNdArray_copy end\n");
    return ret;
}

static PyObject * PyGpuNdArray_Copy(PyGpuNdArrayObject * self, NPY_ORDER order)
{
    DPRINTF("PyGpuNdArray_Copy start\n");
    PyObject * rval = PyGpuNdArray_New();
    //TODO find how to refcount descr.
    PyGpuNdArray_DESCR(rval) = PyGpuNdArray_DESCR(self);
    if ((!rval) || (-1 == PyGpuNdArray_NDIM(self))) {
        return rval;
    }
    if (PyGpuNdArray_alloc_contiguous((PyGpuNdArrayObject*)rval,
                                      PyGpuNdArray_NDIM(self),
                                      PyGpuNdArray_DIMS(self),
                                      order)) {
        Py_DECREF(rval);
        return NULL;
    }

    if (PyGpuNdArray_CopyFromPyGpuNdArray((PyGpuNdArrayObject*)rval, self)) {
        Py_DECREF(rval);
        return NULL;
    }
    if (order == NPY_F_CONTIGUOUS)
        PyGpuNdArray_FLAGS(self) |= NPY_F_CONTIGUOUS;

#ifdef DEBUG
    PyGpuNdArray_fprint(stderr, self);
    PyGpuNdArray_fprint(stderr, (PyGpuNdArrayObject *)rval);
#endif
    DPRINTF("PyGpuNdArray_Copy end\n");
    return rval;
}

PyObject * PyGpuNdArray_DeepCopy(PyGpuNdArrayObject * self, PyObject * memo)
{
    assert(PyDict_Check(memo));
    PyObject * selfkey = PyInt_FromLong((long)self);
    assert(selfkey);

    if (PyDict_Contains(memo, selfkey)) {
        PyObject * rval = PyDict_GetItem(memo, selfkey);
        Py_DECREF(selfkey);
        Py_XINCREF(rval);
        return rval;
    } else {
        DPRINTF("PyGpuNdArray_DeepCopy: startd deepcopy\n");
        PyObject * rval = PyGpuNdArray_Copy(self);
        if (NULL == rval) {
            Py_DECREF(selfkey);
            return NULL;
        }

        DPRINTF("DeepCopy created %p\n", rval);
        DPRINTF("DeepCopy created %p %p\n", PyGpuNdArray_DESCR(rval), PyGpuNdArray_DATA(rval));
        if (PyDict_SetItem(memo, selfkey, rval)) {
            Py_DECREF(rval);
            Py_DECREF(selfkey);
            return NULL;
        }
        Py_DECREF(selfkey);
        DPRINTF("PyGpuNdArray_DeepCopy: startd end\n");
        return rval;
    }
}

PyObject * PyGpuNdArray_View(PyGpuNdArrayObject * self)
{
    PyGpuNdArrayObject * rval = (PyGpuNdArrayObject*)PyGpuNdArray_New(PyGpuNdArray_NDIM(self));
    if (!rval || PyGpuNdArray_set_data(rval, PyGpuNdArray_DATA(self),
                              (PyObject *)self, PyGpuNdArray_OFFSET(self))) {
        Py_XDECREF(rval);
        DPRINTF("PyGpuNdArray_View: no rval or PyGpuNdArray_set_data "
                "failed: self=%p, rval=%p rval_base=%p\n",
                self, rval, rval->base);
        return NULL;
    } else {
        for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i) {
            PyGpuNdArray_DIM(rval, i) = PyGpuNdArray_DIMS(self)[i];
            PyGpuNdArray_STRIDE(rval, i) = PyGpuNdArray_STRIDES(self)[i];
        }
    }
    DPRINTF("PyGpuNdArray_View: self=%p, self->base=%p"
            " rval=%p rval->base=%p\n",
            self, self->base, rval, rval->base);
    //TODO: find how to refcount on the descr!
    //Py_INCREF(PyGpuNdArray_DESCR(self));
    PyGpuNdArray_DESCR(rval) = PyGpuNdArray_DESCR(self);
    PyGpuNdArray_FLAGS(rval) = PyGpuNdArray_FLAGS(self);
    PyGpuNdArray_FLAGS(rval) &= ~NPY_OWNDATA;
    

    return (PyObject*)rval;
}

//updated for offset
PyObject * PyGpuNdArray_CreateArrayObj(PyGpuNdArrayObject * self)
{
    DPRINTF("PyGpuNdArray_CreateArrayObj\n");

    if(PyGpuNdArray_NDIM(self)>=0 && PyGpuNdArray_SIZE(self)==0){
      npy_intp * npydims = (npy_intp*)malloc(PyGpuNdArray_NDIM(self) * sizeof(npy_intp));
      assert (npydims);
      for (int i = 0; i < PyGpuNdArray_NDIM(self); ++i)
	npydims[i] = (npy_intp)(PyGpuNdArray_DIMS(self)[i]);

      // Numpy will do a decref on the description.
      Py_INCREF(PyGpuNdArray_DESCR(self));

      // We can't use PyArray_{Empty,EMPTY} as they segfault when size == 0
      PyObject * rval = PyArray_NewFromDescr(&PyArray_Type,
                                             PyGpuNdArray_DESCR(self),
                                             PyGpuNdArray_NDIM(self),
                                             npydims,
                                             NULL,
                                             NULL,
                                             0,
                                             NULL);

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
        DPRINTF("PyGpuNdArray_CreateArrayObj: gpu array already contiguous %p\n", contiguous_self);
        //}else if(PyGpuNdArray_ISONESEGMENT(self)){
        //TODO implement special object handling to speed up transfer
        //  DPRINTF("CreateArrayObj one segment, with special handling %p\n", contiguous_self);
        //PyErr_SetString(PyExc_ValueError, "PyGpuNdArray_CreateArrayObj: Need PyGpuNdArray_Copy or some other nd array mandling to transfer contiguous bloc with negative stride.");
        //return NULL;
    } else {
        contiguous_self = (PyGpuNdArrayObject*)PyGpuNdArray_Copy(self);
        DPRINTF("CreateArrayObj created contiguous %p\n", contiguous_self);
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

    int err = PyGpuMemcpy(PyArray_DATA(rval),
                          PyGpuNdArray_DATA(contiguous_self),
                          PyGpuNdArray_OFFSET(contiguous_self),
                          PyArray_SIZE(rval) * PyArray_ITEMSIZE(rval),
                          PyGpuDeviceToHost);
    if (err) {
        Py_DECREF(contiguous_self);
        Py_DECREF(rval);
        rval = NULL;
    }
    Py_DECREF(contiguous_self);
    return rval;
}

static PyObject *
PyGpuNdArray_Empty(int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran)
{
    DPRINTF("PyGpuNdArray_Empty: start!\n");
    PyGpuNdArrayObject* rval = (PyGpuNdArrayObject*)PyGpuNdArray_New();
    PyGpuNdArray_DESCR(rval) = dtype;
    if (!rval) {
        DPRINTF("PyGpuNdArray_Empty: fail!\n");
        return NULL;
    }
    NPY_ORDER order = NPY_CORDER;
    if (fortran!=0)
        order = NPY_FORTRANORDER;

    if (PyGpuNdArray_alloc_contiguous(rval, nd, dims, order)) {
        Py_DECREF(rval);
        return NULL;
    }

    DPRINTF("PyGpuNdArray_Empty: end!\n");
    return (PyObject*) rval;
}

//DONE: dtype, offset not needed, flags
static PyObject * 
PyGpuNdArray_Zeros(int nd, npy_intp* dims, PyArray_Descr* dtype, int fortran)
{
    DPRINTF("PyGpuNdArray_Zeros: start!\n");
    PyObject * rval = PyGpuNdArray_Empty(nd, dims, dtype, fortran);
    if (!rval) {
        return rval;
    }

    int total_elements = 1;
    for(int i=0;i<nd;i++)
      total_elements*=dims[i];

    // total_elements now contains the size of the array, in reals
    int total_size = total_elements * dtype->elsize;
    
    // Fill with zeros
    int err = PyGpuMemset(PyGpuNdArray_DATA(rval), 0, total_size);
    if (err) {
        Py_DECREF(rval);
        return NULL;
    }

    DPRINTF("PyGpuNdArray_Zeros: end!\n");
    return (PyObject*) rval;
}

// declared as a static method (hence "dummy" is not used)
// numpy.zeros(shape, dtype=float, order='C')
static PyObject * 
PyGpuNdArray_zeros(PyObject* dummy, PyObject* args, PyObject *kargs)
{
    static const char *kwlist[] = {"shape","dtype","order",NULL}; /* XXX ? */
    PyArray_Descr *typecode = NULL;
    PyObject * shape = NULL;
    NPY_ORDER order = PyArray_CORDER;
    bool fortran = false;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kargs, "O|O&O&",
                                     (char**)kwlist,
	                             &shape,
                                     PyArray_DescrConverter,
                                     &typecode,
                                     PyArray_OrderConverter,
                                     &order)) {
        Py_XDECREF(typecode);
        Py_XDECREF(shape);
        return ret;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = true;
    }
    else {
        fortran = false;
    }

    if(!PySequence_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    if (!typecode)
        typecode = PyArray_DescrFromType(NPY_FLOAT64);

    int shplen = PySequence_Length(shape);

    if (shplen == 0)
    {
        return PyGpuNdArray_Zeros(0, NULL, typecode, fortran);
    }

    npy_intp* newdims = (npy_intp *)malloc(sizeof(npy_intp) * shplen);

    if (!newdims)
    {
        PyErr_SetString(PyExc_MemoryError,
            "PyGpuNdArray_Zeros: Failed to allocate temporary space");
        return NULL;
    }

    // start from the end to compute strides
    for (int i = shplen-1; i >= 0; --i)
    {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if(shp_el_obj == NULL)
        {
            // shouldn't happen since we checked length before...
            PyErr_SetString(PyExc_RuntimeError, "PyGpuNdArray_Zeros: Index out of bound in sequence");
            free(newdims);
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        newdims[i] = shp_el;
    }

    PyObject* rval = PyGpuNdArray_Zeros(shplen, newdims, typecode, fortran);

    free(newdims);

    return (PyObject*)rval;
}

// declared as a static method (hence "dummy" is not used)
// numpy.empty(shape, dtype=float, order='C')
static PyObject * 
PyGpuNdArray_empty(PyObject* dummy, PyObject* args, PyObject *kargs)
{
    static const char *kwlist[] = {"shape","dtype","order",NULL}; /* XXX ? */
    PyArray_Descr *typecode = NULL;
    PyObject * shape = NULL;
    NPY_ORDER order = PyArray_CORDER;
    bool fortran = false;
    PyObject *ret = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kargs, "O|O&O&",
                                     (char **)kwlist,
	                             &shape,
                                     PyArray_DescrConverter,
                                     &typecode,
                                     PyArray_OrderConverter,
                                     &order)) {
        Py_XDECREF(typecode);
        Py_XDECREF(shape);
        return ret;
    }
    if (order == PyArray_FORTRANORDER) {
        fortran = true;
    }
    else {
        fortran = false;
    }

    if(!PySequence_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    if (!typecode)
        typecode = PyArray_DescrFromType(NPY_FLOAT64);

    int shplen = PySequence_Length(shape);

    if (shplen == 0)
    {
        return PyGpuNdArray_Empty(0, NULL, typecode, fortran);
    }

    npy_intp* newdims = (npy_intp *)malloc(sizeof(npy_intp) * shplen);

    if (!newdims)
    {
        PyErr_SetString(PyExc_MemoryError,
            "PyGpuNdArray_empty: Failed to allocate temporary space");
        return NULL;
    }

    // start from the end to compute strides
    for (int i = shplen-1; i >= 0; --i)
    {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if(shp_el_obj == NULL)
        {
            // shouldn't happen since we checked length before...
            PyErr_SetString(PyExc_RuntimeError, "PyGpuNdArray_empty: Index out of bound in sequence");
            free(newdims);
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        newdims[i] = shp_el;
    }

    PyObject* rval = PyGpuNdArray_Empty(shplen, newdims, typecode, fortran);

    free(newdims);

    return (PyObject*)rval;
}

static PyMethodDef PyGpuNdArray_methods[] =
{
    {"__array__",
        (PyCFunction)PyGpuNdArray_CreateArrayObj, METH_NOARGS,
        "Copy from the device to a numpy ndarray"},
    {"copy",
     (PyCFunction)PyGpuNdArray_copy, METH_VARARGS|METH_KEYWORDS,
        "Create a deep copy of this object."},
    {"view",
        (PyCFunction)PyGpuNdArray_View, METH_NOARGS,
        "Create a view of this object."},
    {"__copy__",
        (PyCFunction)PyGpuNdArray_Copy, METH_NOARGS,
        "Create a copy of this object as numpy does. Why numpy do a copy of the data when the object is a view?"},
    {"__deepcopy__",
        (PyCFunction)PyGpuNdArray_DeepCopy, METH_O,
        "Create a copy of this object"},
/*
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
    DPRINTF("PyGpuNdArray_get_shape\n");

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
    return (PyObject *)PyInt_FromLong(PyGpuNdArray_ITEMSIZE(self));
}

static PyGetSetDef PyGpuNdArray_getset[] = {
    {(char*)"base",
        (getter)PyGpuNdArray_get_base,
        NULL,
        (char*)"Return the object stored in the base attribute",
        NULL},
    {(char*)"bytes",
        (getter)PyGpuNdArray_get_data,
        NULL,
        (char*)"device data pointer",
        NULL},
    {(char*)"shape",
        (getter)PyGpuNdArray_get_shape,
        (setter)PyGpuNdArray_set_shape,
        (char*)"shape of this ndarray (tuple)",
        NULL},
    {(char*)"strides",
        (getter)PyGpuNdArray_get_strides,
        NULL,//(setter)PyGpuNdArray_set_strides,
        (char*)"data pointer strides (in elements)",
        NULL},
    {(char*)"ndim",
        (getter)PyGpuNdArray_get_ndim,
        NULL,
        (char*)"The number of dimensions in this object",
        NULL},
    {(char*)"offset",
        (getter)PyGpuNdArray_get_offset,
        NULL,
        (char*)"Return the offset value",
        NULL},
    {(char*)"size",
        (getter)PyGpuNdArray_get_size,
        NULL,
        (char*)"The number of elements in this object.",
        NULL},
    {(char*)"data_allocated",
        (getter)PyGpuNdArray_get_data_allocated,
        NULL,
        (char*)"The size of the allocated memory on the device.",
        NULL},
    {(char*)"itemsize",
        (getter)PyGpuNdArray_get_itemsize,
        NULL,
        (char*)"The size of the base element.",
        NULL},
    {(char*)"dtype",
	(getter)PyGpuNdArray_get_dtype,
	NULL,
        (char*)"The dtype of the element",
	NULL},
    {(char*)"flags",
        (getter)PyGpuNdArray_get_flags,
        NULL,
        (char*)"Return the flags as a dictionary",
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
PyGpuNdArray_add_offset(PyGpuNdArrayObject * self, int offset)
{
    DPRINTF("PyGpuNdArray_add_offset: %p %d\n", self, offset);

#if OFFSET
    PyGpuNdArray_OFFSET(self) += offset;
#else
    PyGpuNdArray_DATA(self) += offset;
#endif
    return 0;
}


static int
PyGpuNdArray_set_data(PyGpuNdArrayObject * self, char * data, PyObject * base, int offset)
{
    DPRINTF("PyGpuNdArray_set_data: %p %p %p %d\n", self, data, base, offset);
    if (self->data_allocated)
    {
        assert(PyGpuNdArray_DATA(self));
        if (device_free(PyGpuNdArray_DATA(self)))
        {
            PyGpuNdArray_DATA(self) = NULL;
            self->data_allocated = 0;
            DPRINTF("PyGpuNdArray_set_data: device_free failed!\n");
            PyErr_SetString(PyExc_ValueError, "PyGpuNdArray_set_data: device_free failed");
            return -1;
        }
    }

    // Get the original base object (base.base.base...)
    // TODO: check that base is indeed a CudaNdarray?
    PyObject * orig_base = base;
    // base is not always a PyGpuNdArrayObject. It can be a GpuArray from pycuda, ...
    while (orig_base && PyGpuNdArray_Check(orig_base) && ((PyGpuNdArrayObject*) orig_base)->base)
    {
        // base_base is itself a view
        orig_base = ((PyGpuNdArrayObject*) orig_base)->base;
    }

    //N.B. XDECREF and XINCREF are no-ops for NULL pointers
    if (PyGpuNdArray_BASE(self) != orig_base)
    {
        Py_XDECREF(PyGpuNdArray_BASE(self));
        PyGpuNdArray_BASE(self) = orig_base;
        Py_XINCREF(PyGpuNdArray_BASE(self));
    }
    self->data_allocated = 0;
#if OFFSET
    PyGpuNdArray_DATA(self) = data;
    PyGpuNdArray_OFFSET(self) = offset;
#else
    PyGpuNdArray_DATA(self) = data + offset;
#endif

    return 0;
}

// Will by called by __getitem__ in Python
static PyObject *
PyGpuNdArray_Subscript(PyObject * py_self, PyObject * key)
{
    DPRINTF("Subscript start\n");
    PyGpuNdArrayObject * self = (PyGpuNdArrayObject*) py_self;
    PyObject * py_rval = NULL;
    PyGpuNdArrayObject * rval = NULL;
    PyObject * intobj = NULL;

    //PyObject_Print(key, stderr, 0);

    if (key == Py_Ellipsis)
    {
        DPRINTF("Subscript with ellipse \n");
        Py_INCREF(py_self);
        DPRINTF("Subscript with ellipse end\n");
        return py_self;
    }
    if ((intobj=PyNumber_Int(key))) //INDEXING BY INTEGER
    {
#ifdef DEBUG
        PyGpuNdArray_fprint(stderr, self);
#endif
        DPRINTF("Subscript with int \n");

        int d_idx = PyInt_AsLong(intobj);
        Py_DECREF(intobj); intobj=NULL;

        DPRINTF("Subscript with int 1\n");
        if (PyGpuNdArray_NDIM(self) == 0) {
            PyErr_SetString(PyExc_IndexError, "0-d arrays can't be indexed");
            return NULL;
        }else if (PyGpuNdArray_NDIM(self)< 0){
            PyErr_SetString(PyExc_IndexError, "nd arrays must have a number of dim > 0!");
            return NULL;
        }
        int d_dim = PyGpuNdArray_DIMS(self)[0];
        int offset = 0;
        DPRINTF("Subscript with int 2\n");

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
        DPRINTF("Subscript with int 3\n");

        //Add the original offset
        offset += PyGpuNdArray_OFFSET(self);

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(PyGpuNdArray_NDIM(self) - 1);
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);

        DPRINTF("Subscript with int 4\n");
        //initialize the view's data pointer to our own.
        assert (0 == rval->data_allocated);
        if (PyGpuNdArray_set_data(rval, PyGpuNdArray_DATA(self), (PyObject *) self, offset)){
            Py_DECREF(rval);
            return NULL;
        }
        DPRINTF("Subscript with int 5\n");

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
        DPRINTF("Subscript with slice \n");
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

        DPRINTF("start %zd\nstop %zd\n step %zd\n slen %zd\n",
                start, stop, step, slen);

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(PyGpuNdArray_NDIM(self));
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);
        assert (0 == rval->data_allocated);
        if (PyGpuNdArray_set_data(rval,
                                  PyGpuNdArray_DATA(self),
                                  py_self,
                                  start * PyGpuNdArray_STRIDE(self, 0)
                                  + PyGpuNdArray_OFFSET(self))) {
            Py_DECREF(rval);
            return NULL;
        }

        //initialize dimension 0 of rval
        PyGpuNdArray_STRIDE(rval, 0) = step * PyGpuNdArray_STRIDES(self)[0];
        PyGpuNdArray_DIM(rval, 0) = slen;
        DPRINTF("rval stride %zd\n", PyGpuNdArray_STRIDES(rval)[0]);
        // initialize dimensions > 0 of rval
        for (int d = 1; d < PyGpuNdArray_NDIM(self); ++d) {
            PyGpuNdArray_STRIDE(rval, d) = PyGpuNdArray_STRIDES(self)[d];
            PyGpuNdArray_DIM(rval, d) = PyGpuNdArray_DIMS(self)[d];
        }
    }
    if (PyTuple_Check(key)) //INDEXING BY TUPLE
    {
        DPRINTF("Subscript with tuple \n");
        //elements of the tuple can be either integers or slices
        //the dimensionality of the view we will return is diminished for each slice in the tuple
        int tuple_start_index = 0;
        if (PyTuple_Size(key) > PyGpuNdArray_NDIM(self))
        {
            if (PyTuple_GetItem(key, 0) == Py_Ellipsis &&
                PyTuple_Size(key) == PyGpuNdArray_NDIM(self) + 1)
            {
                tuple_start_index = 1;
                DPRINTF("Subscript with tuple staring with an extra ellipse"
                        " at the start.\n");
            }
            else{
                PyErr_SetString(PyExc_IndexError,
                                "index error, specified more dimensions then"
                                " the number of existing dimensions");
                return NULL;
            }
        }

        //calculate the number of dimensions in the return value
        int rval_nd = PyGpuNdArray_NDIM(self);
        for (int tuple_d = tuple_start_index; tuple_d < PyTuple_Size(key);
             ++tuple_d)
        {
            //On some paltform PyInt_Check(<type 'numpy.int64'>) return true, other it return false.
            //So we use PyArray_IsAnyScalar that should covert everything.
            rval_nd -= PyArray_IsAnyScalar(PyTuple_GetItem(key, tuple_d));
        }

        //allocate our subtensor view
        py_rval = PyGpuNdArray_New(rval_nd);
        rval = (PyGpuNdArrayObject*) py_rval;
        if (!rval) return NULL;
        assert (0 == rval->data_allocated);

        //TODO: find how to refcount on the descr!
        PyGpuNdArray_DESCR(py_rval) = PyGpuNdArray_DESCR(self);

        //initialize the view's data pointer to our own.
        if (PyGpuNdArray_set_data(rval, PyGpuNdArray_DATA(self),
                                  py_self, PyGpuNdArray_OFFSET(self)))
        {
            Py_DECREF(rval);
            return NULL;
        }

        // rval_d will refer to the current dimension in the rval.
        // It will not be incremented for integer keys, but will be incremented for slice
        // keys
        int rval_d = 0;

        for (int self_d = 0, tuple_d = tuple_start_index;
             self_d < PyGpuNdArray_NDIM(self); ++self_d, ++tuple_d)
        {
            // keys can be shorter than PyGpuNdArray_NDIM(self).
            // when that happens, it means that the remaining dimensions are "full slices"
            if (tuple_d >= PyTuple_Size(key))
            {
                PyGpuNdArray_STRIDE(rval, rval_d) =
                    PyGpuNdArray_STRIDES(self)[tuple_d];
                PyGpuNdArray_DIM(rval, rval_d) =
                    PyGpuNdArray_DIMS(self)[tuple_d];
                ++rval_d;
                DPRINTF("Subscript extra dims to append %zd %zd\n",
                        PyGpuNdArray_STRIDE(rval, rval_d),
                        PyGpuNdArray_DIM(rval, rval_d));
            }
            else
            {
                PyObject * key_d = PyTuple_GetItem(key, tuple_d);

                if (PySlice_Check(key_d))
                {
                    Py_ssize_t start, stop, step, slen;
                    if (PySlice_GetIndicesEx((PySliceObject*)key_d,
                                             PyGpuNdArray_DIMS(self)[self_d],
                                             &start, &stop, &step, &slen))
                    {
                        Py_DECREF(rval);
                        return NULL;
                    }
                    PyGpuNdArray_add_offset(rval, start * PyGpuNdArray_STRIDES(self)[self_d]);
                    PyGpuNdArray_STRIDE(rval, rval_d) = step * PyGpuNdArray_STRIDES(self)[self_d];
                    PyGpuNdArray_DIM(rval, rval_d) = slen;

                    DPRINTF("rval_d %d self_d %d\n start %zd\nstop %zd\n step %zd\n slen %zd\n",
                            rval_d, self_d, start, stop, step, slen);
                    ++rval_d;
                }
                else if ((intobj=PyNumber_Int(key_d)))
                {
                    assert(PyArray_IsAnyScalar(key_d));
                    int d_idx = PyInt_AsLong(intobj);
                    Py_DECREF(intobj);
                    intobj = NULL;
                    int d_dim = PyGpuNdArray_DIMS(self)[self_d];

                    if ((d_idx >= 0) && (d_idx < d_dim))
                    {
                        //normal indexing
                        PyGpuNdArray_add_offset(rval, d_idx * PyGpuNdArray_STRIDES(self)[self_d]);
                    }
                    else if ((d_idx < 0) && (d_idx >= -d_dim))
                    {
                        //end-based indexing
                        PyGpuNdArray_add_offset(rval, (d_dim + d_idx) * PyGpuNdArray_STRIDES(self)[self_d]);
                    }
                    else
                    {
                        PyErr_SetString(PyExc_IndexError, "index out of bounds");
                        Py_DECREF(rval);
                        return NULL;
                    }
                }
                else if (key_d == Py_Ellipsis)
                {
                    if (self_d != 0){
                        PyErr_Format(PyExc_IndexError,
                                     "Ellipsis supported only at the start of"
                                     " the tuple");
                        Py_DECREF(rval);
                        return NULL;
                    }
                    DPRINTF("Substript with tuple with the first element an ellipse\n");
                    for( ; self_d < (rval_nd - PyTuple_Size(key) + 1); self_d++)
                    {
                        PyGpuNdArray_STRIDE(rval, rval_d) =
                            PyGpuNdArray_STRIDES(self)[self_d];
                        PyGpuNdArray_DIM(rval, rval_d) =
                            PyGpuNdArray_DIMS(self)[self_d];
                        DPRINTF("Ellipse append dimensions self_%d with %zd %zd\n",
                                self_d,
                                PyGpuNdArray_STRIDE(rval, rval_d),
                                PyGpuNdArray_DIM(rval, rval_d));
                        ++rval_d;
                    }
                    tuple_start_index = 1;
                    self_d--;
                }
                else
                {
                    PyErr_Clear(); // clear the error set by PyNumber_Int
                    PyErr_Format(PyExc_IndexError,
                                 "index must be either int or slice. Got %s",
                                 PyString_AsString(PyObject_Str(key_d)));
                    Py_DECREF(rval);
                    return NULL;
                }
            }
        }
    }
    if (py_rval)
    {
#ifdef DEBUG
        PyGpuNdArray_fprint(stderr, self);
        PyGpuNdArray_fprint(stderr, rval);
#endif
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

    DPRINTF("Subscript end\n");
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
    DPRINTF("PyGpuNdArray_New start\n");
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
    DPRINTF("PyGpuNdArray_New end\n");
    return (PyObject *)self;
}

int
PyGpuNdArray_Check(const PyObject * ob)
{
    DPRINTF("PyGpuNdArray_Check\n");
    //TODO: doesn't work with inheritance
    return PyGpuNdArray_CheckExact(ob);
}
int
PyGpuNdArray_CheckExact(const PyObject * ob)
{
    DPRINTF("PyGpuNdArray_CheckExact\n");
    return ((ob->ob_type == &PyGpuNdArrayType) ? 1 : 0);
}

static PyObject *
PyGpuNdArray_as_c_contiguous(PyObject* dummy, PyObject* args, PyObject *kargs)
{
    DPRINTF("PyGpuNdArray_as_c_contiguous:start\n");
    static const char *kwlist[] = {"a", "dtype", NULL};
    PyArray_Descr *typecode = NULL;
    PyObject *self_ = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kargs, "O|O&",
                                     (char **)kwlist,
                                     &self_,
                                     PyArray_DescrConverter,
                                     &typecode)) {
        Py_XDECREF(typecode);
        Py_XDECREF(self_);
        return NULL;
    }
    assert(typecode == NULL);
    if (!PyGpuNdArray_Check(self_)){
        PyErr_SetString(PyExc_TypeError,
                        "PyGpuNdArray_as_c_contiguous:"
                        " PyGpuNdArrayObject required");
        return NULL;
    }

    PyGpuNdArrayObject *self = (PyGpuNdArrayObject*)self_;
    if (PyGpuNdArray_is_c_contiguous(self)){
        Py_INCREF(self);
        if (PyGpuNdArray_NDIM(self) == 0){
            //numpy.ascontiguous() always return object with 1d.
            DPRINTF("PyGpuNdArray_as_c_contiguous: upcast to 1d tensor end\n");
            PyObject * rval = PyGpuNdArray_View(self);
            if (!rval)
                return NULL;
            PyGpuNdArray_set_nd((PyGpuNdArrayObject*)rval, 1);
            PyGpuNdArray_DIM(rval, 0) = 1;
            PyGpuNdArray_STRIDE(rval, 0) = PyGpuNdArray_ITEMSIZE(rval);
            return rval;
        }
        DPRINTF("PyGpuNdArray_as_c_contiguous: no copy end\n");
        return (PyObject*)self;
    }

    PyObject * ret = PyGpuNdArray_Copy(self);
    DPRINTF("PyGpuNdArray_as_c_contiguous: copy end\n");
    return ret;
}
static PyObject *
PyGpuNdArray_as_f_contiguous(PyObject* dummy, PyObject* args, PyObject *kargs)
{
    DPRINTF("PyGpuNdArray_as_f_contiguous:start\n");
    static const char *kwlist[] = {"a", "dtype", NULL};
    PyArray_Descr *typecode = NULL;
    PyObject *self_ = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kargs, "O|O&",
                                     (char **)kwlist,
                                     &self_,
                                     PyArray_DescrConverter,
                                     &typecode)) {
        Py_XDECREF(typecode);
        Py_XDECREF(self_);
        return NULL;
    }
    assert(typecode == NULL);
    if (!PyGpuNdArray_Check(self_)){
        PyErr_SetString(PyExc_TypeError,
                        "PyGpuNdArray_as_f_contiguous:"
                        " PyGpuNdArrayObject required");
        return NULL;
    }

    PyGpuNdArrayObject *self = (PyGpuNdArrayObject*)self_;
    if (PyGpuNdArray_is_f_contiguous(self)){
        Py_INCREF(self);
        if (PyGpuNdArray_NDIM(self) == 0){
            //numpy.ascontiguous() always return object with 1d.
            PyObject * rval = PyGpuNdArray_View(self);
            if (!rval)
                return NULL;
            PyGpuNdArray_set_nd((PyGpuNdArrayObject*)rval, 1);
            PyGpuNdArray_DIM(rval, 0) = 1;
            PyGpuNdArray_STRIDE(rval, 0) = PyGpuNdArray_ITEMSIZE(rval);
            DPRINTF("PyGpuNdArray_as_f_contiguous: upcast to 1d tensor end\n");
            return rval;
        }
        DPRINTF("PyGpuNdArray_as_f_contiguous: no copy end\n");
        return (PyObject*)self;
    }

    PyObject * ret = PyGpuNdArray_Copy(self, NPY_FORTRANORDER);
    DPRINTF("PyGpuNdArray_as_f_contiguous: copy end\n");
    return ret;
}

#ifdef WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
extern void setup_context(cl_context c);

PyObject *
PyGpuNdArray_set_opencl_context(PyObject *mod, PyObject *ctx) {
    Py_ssize_t v;

    v = PyInt_AsSsize_t(ctx);
    if (v == -1 && PyErr_Occurred())
        return NULL;

    setup_context((cl_context)v);
    
    Py_INCREF(Py_None);
    return Py_None;
}
#endif

static PyMethodDef module_methods[] = {
    //{"dimshuffle", PyGpuNdArray_Dimshuffle, METH_VARARGS, "Returns the dimshuffle of a PyGpuNdArray."},
    {"outstanding_mallocs", outstanding_mallocs, METH_VARARGS, "how many more mallocs have been called than free's"},
    {"zeros",
       (PyCFunction)PyGpuNdArray_zeros, METH_VARARGS|METH_KEYWORDS,
       "Create a new PyGpuNdArray with specified shape, filled with zeros."},
    {"empty",
       (PyCFunction)PyGpuNdArray_empty, METH_VARARGS|METH_KEYWORDS,
       "Create a new PyGpuNdArray with specified shape, filled with zeros."},
    {"ascontiguousarray",
       (PyCFunction)PyGpuNdArray_as_c_contiguous, METH_VARARGS|METH_KEYWORDS,
       "If the array is not c contiguous, copy it to a new c contiguous region."},
    {"asfortranarray",
       (PyCFunction)PyGpuNdArray_as_f_contiguous, METH_VARARGS|METH_KEYWORDS,
       "If the array is not f contiguous, copy it to a new c contiguous region."},
#ifdef WITH_OPENCL
    {"set_opencl_context",
     PyGpuNdArray_set_opencl_context, METH_O,
     "Set the OpenCL context to use for allocations and work."},
#endif
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
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
