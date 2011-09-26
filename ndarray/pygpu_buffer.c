#include <Python.h>

#include "compyte_buffer.h"

typedef struct _pygpu_buffer_ops {
  PyObject_HEAD
  compyte_buffer_ops *ops;
} pygpu_buffer_ops;

static PyObject *Op_alloc(pygpu_buffer_ops *self,
			  PyObject *args, PyObject *kwds)
{
  Py_ssize_t size;
  gpubuf res;
  static const char *kwlist[] = {"size", NULL};
  
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwdlist, &size))
    return -1;
  res = self.ops->buffer_alloc(size);
  
}

static PyMethodDef ops_methods[] = {
  {NULL, NULL},
};

static PyTypeObject buffer_ops_Type = {
  PyObject_HEAD_INIT(NULL)
  0,                         /*ob_size*/
  "pygpu.buffer.ops",        /*tp_name*/
  sizeof(noddy_NoddyObject), /*tp_basicsize*/
  0,                         /*tp_itemsize*/
  0,                         /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,        /*tp_flags*/
  "Buffer ops",           /* tp_doc */
};

static PyObject *buffer_ops_NEW(compyte_buffer_ops *ops)
{
  pygpu_buffer_ops *res = NULL;
  res = PyObject_NEW(pygpu_buffer_ops, &pygpu_buffer_ops_Type);
  if (res != NULL) {
    res->ops = ops;
  }
  return (PyObject *)res;
}

static PyMethodDef mod_methods[] = {
  {NULL, NULL},
};

#ifndef PyMODINIT_FUNC/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINTI_FUNC initpygpu_buffer()
{
  PyObject *m;
  PyObject *ops;
  
  if (PyType_Ready(&noddy_NoddyType) < 0)
    return;
  
  m = Py_InitModule("pygpu.buffer", methods);

#ifdef WITH_CUDA
  ops = buffer_ops_NEW(cuda_ops);
  if (ops != NULL) {
    PyModule_AddObject(m, "cuda_ops", ops);
    Py_DECREF(ops);
  }
#endif
#ifdef WITH_OPENCL
  ops = buffer_ops_NEW(opencl_ops);
  if (ops != NULL) {
    PyModule_AddObject(m, "opencl_ops", ops);
    Py_DECREF(ops);
  }
#endif
}
