#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>
#include <iostream>

#include "pygpu_ndarray.cuh"

//If true, when there is a gpu malloc or free error, we print the size of allocated memory on the device.
#define COMPUTE_GPU_MEM_USED 0
#define VERBOSE_ALLOC_FREE 0
//If true, we fill with NAN allocated device memory.
#define ALLOC_MEMSET 0

/////////////////////////
// Alloc and Free
/////////////////////////
int _outstanding_mallocs[] = {0,0};
#if COMPUTE_GPU_MEM_USED
int _allocated_size = 0;
const int TABLE_SIZE = 10000;
struct table_struct{
  void* ptr;
  int size;
};
table_struct _alloc_size_table[TABLE_SIZE];
#endif
void * device_malloc(size_t size)
{
    void * rval=NULL;
    cudaError_t err = cudaMalloc(&rval, size);
    if (cudaSuccess != err){
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error allocating %li bytes of device memory (%s). %d already allocated\n",
		(long)size, cudaGetErrorString(err),_allocated_size);
#else
        fprintf(stderr, "Error allocating %li bytes of device memory (%s).\n",
		(long)size, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "Error allocating %li bytes of device memory (%s).",
		     (long)size, cudaGetErrorString(err));
        return NULL;
    }
    _outstanding_mallocs[0] += (rval != NULL);
#if COMPUTE_GPU_MEM_USED
    for(int i=0;i<TABLE_SIZE;i++){
      if(NULL==_alloc_size_table[i].ptr){
        _alloc_size_table[i].ptr=rval;
        _alloc_size_table[i].size=size;
        break;
      }
    }
    _allocated_size += size;
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "allocated %li bytes of device memory (%s). %d already allocated, ptr: %p\n",
	      (long)size, cudaGetErrorString(err),_allocated_size,rval);
#else
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "allocated %li bytes of device memory (%s). ptr: %p\n",
	      (long)size, cudaGetErrorString(err),rval);

#endif

    if(ALLOC_MEMSET){
      //We init them to nan to make sure we catch more debug case.
      cudaMemset(rval, 0xFF, size);
      //printf("MEMSET\n");
    }
    return rval;
}

int device_free(void *ptr)
{
    // if there is no gpu context, the call to cudaFree will fail; skip it entirely
    /*if(!g_gpu_context_active){
        return 0;
	}*/
    cudaError_t err =  cudaFree(ptr);
    if (cudaSuccess != err){
#if COMPUTE_GPU_MEM_USED
        fprintf(stderr, "Error freeing device pointer %p (%s).%d byte already allocated\n",
		ptr, cudaGetErrorString(err), _allocated_size);
#else
        fprintf(stderr, "Error freeing device pointer %p (%s).\n",
		ptr, cudaGetErrorString(err));
#endif
        PyErr_Format(PyExc_MemoryError, "error freeing device pointer %p (%s)",
		     ptr, cudaGetErrorString(err));
        return -1;
    }
    _outstanding_mallocs[0] -= (ptr != NULL);
#if COMPUTE_GPU_MEM_USED
    int i=0;
    size_t total_freed = 0;
    for(;i<TABLE_SIZE;i++)
      if(_alloc_size_table[i].ptr==ptr){
        _allocated_size -= _alloc_size_table[i].size;
        total_freed += _alloc_size_table[i].size;
        _alloc_size_table[i].ptr=0;
        _alloc_size_table[i].size=0;

        break;
      }
    if(i==TABLE_SIZE)
      printf("Unallocated unknow size!\n");
    if(VERBOSE_ALLOC_FREE)
      fprintf(stderr, "freed %li bytes of device memory (%s). %d already allocated, ptr=%p\n", (long)total_freed, cudaGetErrorString(err),_allocated_size,ptr);
#endif
    return 0;
}
static PyObject *
outstanding_mallocs(PyObject* self, PyObject * args)
{
    return PyInt_FromLong(_outstanding_mallocs[0]);
}



/////////////////////////
// Static helper methods
/////////////////////////

static void
PyGpuNdArrayObject_null_init(PyGpuNdArrayObject *self)
{
    self->data = NULL;
    self->offset = 0;
    self->nd = -1;
    self->base = NULL;
    self->dimensions = NULL;
    self->strides = NULL;
    self->flags = NPY_DEFAULT;
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
    if (0) std::cerr << "PyGpuNdArrayObject dealloc " << self << '\n';
    if (0) std::cerr << "PyGpuNdArrayObject dealloc " << self << " " << self->data << '\n';

    if(self->ob_refcnt>1)
      printf("WARNING:PyGpuNdArrayObject_dealloc called when their is still active reference to it.\n");

    if (self->data_allocated){
        assert(self->data);
        if (self->data){
            if (device_free(self->data)){
	      fprintf(stderr,
		  "!!!! error freeing device memory %p (self=%p)\n",
		  self->data, self);
	    }
	    self->data = NULL;
	}
    }
    self->offset = 0;
    self->nd = -1;
    Py_XDECREF(self->base);
    self->base = NULL;
    if (self->dimensions){
        free(self->dimensions);
        self->dimensions = NULL;
    }
    if (self->strides){
        free(self->strides);
        self->strides = NULL;
    }
    if (self->strides){
        free(self->strides);
        self->strides = NULL;
    }
    self->flags = NPY_DEFAULT;
    Py_XDECREF(self->descr);
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

static PyObject *
PyGpuNdArray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyGpuNdArrayObject *self;

    self = (PyGpuNdArrayObject *)type->tp_alloc(type, 0);
    if (self != NULL){
        PyGpuNdArrayObject_null_init(self);
        ++_outstanding_mallocs[1];
    }
    return (PyObject *)self;
}

static int
PyGpuNdArray_init(PyGpuNdArrayObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *arr=NULL;

    if (! PyArg_ParseTuple(args, "O", &arr))
        return -1;
    if (! PyArray_Check(arr)){
        PyErr_SetString(PyExc_TypeError, "PyGpuNdArrayObject_init: PyArray or PyGpuNdArrayObject arg required");
        return -1;
    }

    // We must create a new copy of the PyArray_Descr(or this only increment the refcount?)
    PyArray_Descr * type = PyArray_DescrFromType(PyArray_TYPE(arr));
    self->descr = type;
    int rval = PyGpuNdArray_CopyFromArray(self, (PyArrayObject*)arr);
    return rval;
}


int
PyGpuNdArray_CopyFromArray(PyGpuNdArrayObject * self, PyArrayObject*obj)
{
    if(0) fprintf(stderr, "PyGpuNdArray_CopyFromArray: start\n");

    int err = PyGpuNdArray_alloc_contiguous(self, obj->nd, obj->dimensions);
    if (err) {
        return err;
    }


    int typenum = PyArray_TYPE(obj);
    PyObject * py_src = PyArray_ContiguousFromAny((PyObject*)obj, typenum, self->nd, self->nd);
    if(0) fprintf(stderr, "PyGpuNdArray_CopyFromArray: contiguous!\n");
    if (!py_src) {
        return -1;
    }
    cublasSetVector(PyArray_SIZE(py_src),
		    PyArray_ITEMSIZE(obj),
		    PyArray_DATA(py_src), 1,
		    self->data, 1);
    CNDA_THREAD_SYNC;
    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to device memory");
        Py_DECREF(py_src);
        return -1;
    }
    Py_DECREF(py_src);
    return 0;
}

//updated for offset and dtype
PyObject * PyGpuNdArray_CreateArrayObj(PyGpuNdArrayObject * self)
{
    int verbose = 0;

    assert(self->offset==0);//TODO implement when offset is not 0!

    if(self->nd>=0 && PyGpuNdArray_SIZE(self)==0){
      npy_intp * npydims = (npy_intp*)malloc(self->nd * sizeof(npy_intp));
      assert (npydims);
      for (int i = 0; i < self->nd; ++i) npydims[i] = (npy_intp)(PyGpuNdArray_DIMS(self)[i]);
      PyObject * rval = PyArray_SimpleNew(self->nd, npydims, PyGpuNdArray_TYPE(self));
      free(npydims);
      if (!rval){
        return NULL;
      }
      assert (PyArray_ITEMSIZE(rval) == PyGpuNdArray_ITEMSIZE(self));
      return rval;
    }
    if ((self->nd < 0) || (self->data == 0))
    {
        PyErr_SetString(PyExc_ValueError, "can't copy from un-initialized PyGpuNdArray");
        return NULL;
    }
    PyGpuNdArrayObject * contiguous_self = NULL;
    if (PyGpuNdArray_ISCONTIGUOUS(self))
    {
        contiguous_self = self;
        Py_INCREF(contiguous_self);
        if (verbose) std::cerr << "CreateArrayObj already contiguous" << contiguous_self << '\n';
    }
    else
    {
        //TODO implement PyGpuNdArray_Copy
        //contiguous_self = (PyGpuNdArrayObject*)PyGpuNdArray_Copy(self);
        //  if (verbose) std::cerr << "CreateArrayObj created contiguous" << contiguous_self << '\n';
        PyErr_SetString(PyExc_ValueError, "PyGpuNdArray_CreateArrayObj: Need PyGpuNdArray_Copy to be implemented to be able to transfer not contiguous memory block.");
        return NULL;

    }
    if (!contiguous_self)
    {
        return NULL;
    }

    npy_intp * npydims = (npy_intp*)malloc(self->nd * sizeof(npy_intp));
    assert (npydims);
    for (int i = 0; i < self->nd; ++i) npydims[i] = (npy_intp)(PyGpuNdArray_DIMS(self)[i]);
    PyObject * rval = PyArray_SimpleNew(self->nd, npydims, PyGpuNdArray_TYPE(self));
    free(npydims);
    if (!rval)
    {
        Py_DECREF(contiguous_self);
        return NULL;
    }

    cublasGetVector(PyArray_SIZE(rval), PyArray_ITEMSIZE(rval),
            contiguous_self->data, 1,
            PyArray_DATA(rval), 1);
    CNDA_THREAD_SYNC;

    if (CUBLAS_STATUS_SUCCESS != cublasGetError())
    {
        PyErr_SetString(PyExc_RuntimeError, "error copying data to host");
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
    /*    {"__copy__",
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
    if (self->nd < 0)
    {
        PyErr_SetString(PyExc_ValueError, "PyGpuNdArray not initialized");
        return NULL;
    }
    PyObject * rval = PyTuple_New(self->nd);
    for (int i = 0; i < self->nd; ++i)
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

static PyGetSetDef PyGpuNdArray_getset[] = {
    {"shape",
        (getter)PyGpuNdArray_get_shape,
        (setter)PyGpuNdArray_set_shape,
        "shape of this ndarray (tuple)",
        NULL},
    /*
    {"_strides",
        (getter)PyGpuNdArray_get_strides,
        (setter)PyGpuNdArray_set_strides,
        "data pointer strides (in elements)",
        NULL},
    //gpudata is needed to allow calling pycuda fct with PyGpuNdArray input.
    {"gpudata",
        (getter)PyGpuNdArray_get_dev_data,
        NULL,
        "device data pointer",
        NULL},
    {"_dev_data",
        (getter)PyGpuNdArray_get_dev_data,
        (setter)PyGpuNdArray_set_dev_data,
        "device data pointer",
        NULL},
    {"dtype",
        (getter)PyGpuNdArray_get_dtype,
        NULL,
        "The dtype of the element. Now always float32",
        NULL},
    {"size",
        (getter)PyGpuNdArray_SIZE_Object,
        NULL,
        "The number of elements in this object.",
        NULL},
    //mem_size is neede for pycuda.elementwise.ElementwiseKernel Why do they use size and mem_size of the same value?
    {"mem_size",
        (getter)PyGpuNdArray_SIZE_Object,
        NULL,
        "The number of elements in this object.",
        NULL},
    {"ndim",
        (getter)PyGpuNdArray_get_ndim,
        NULL,
        "The number of dimensions in this object.",
        NULL},
    */
    {NULL, NULL, NULL, NULL}  /* Sentinel */
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
    0, //&PyGpuNdArrayObjectMappingMethods,/*tp_as_mapping*/
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

int
PyGpuNdArray_Check(const PyObject * ob)
{
    //TODO: doesn't work with inheritance
    return PyGpuNdArray_CheckExact(ob);
}
int
PyGpuNdArray_CheckExact(const PyObject * ob)
{
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
}
