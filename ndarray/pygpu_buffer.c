#include <Python.h>

#include "compyte_buffer.h"

static PyObject *BufferOpError;

typedef struct _buffer_ops {
  PyObject_HEAD
  compyte_buffer_ops *ops;
} buffer_ops;

static PyTypeObject buffer_ops_Type;
#define buffer_ops_Check(v)  ((v)->ob_type == &buffer_ops_Type)

static PyObject *Op_alloc(PyObject *s, PyObject *args, PyObject *kwds)
{
  buffer_ops *self = (buffer_ops *)s;
  Py_ssize_t size;
  gpubuf data;
  static const char *kwlist[] = {"size", NULL};

  if (!buffer_ops_Check(self)) {
    PyErr_SetString(PyExc_TypeError, "wrong type for self");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwdlist, &size))
    return NULL;
  data = self->ops->buffer_alloc(size);
  if (data == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate device memory.");
    return NULL;
  }
  return PyCObject_FromVoidPtr(data, self->ops->buffer_free);
}

static PyObject *Op_move(PyObject *s, PyObject *args, PyObject *kwds)
{
  buffer_ops *self = (buffer_ops *)s;
  PyCObject *src, *dst;
  Py_ssize_t src_offset, dst_offset, sz;
  int err;
  static const char *kwdlist[] = {"dst", "dst_offset", "src", "src_offset", "sz", NULL};

  if (!buffer_ops_Check(self)) {
    PyErr_SetString(PyExc_TypeError, "wrong type for self");
    return NULL;
  }

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OnOnn", kwdlist,
				   &dst, &dst_offset, &src, &src_offset, &sz))
    return NULL;

  err = self->ops->buffer_move(PyCObject_AsVoidPtr(dst), (size_t)dst_offset,
			       PyCObject_AsVoidPtr(src), (size_t)src_offset,
			       (size_t)sz);
  if (err == -1) {
    PyErr_SetString(BufferOpError, self->ops->buffer_error());
    return NULL;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ops_methods[] = {
  {"alloc", Op_alloc},
  {"move", Op_move},
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
PyMODINIT_FUNC initpygpu_buffer()
{
  PyObject *m;
  PyObject *ops;
  
  if (PyType_Ready(&noddy_NoddyType) < 0)
    return;
  
  m = Py_InitModule("pygpu.buffer", methods);
  BufferOpError = PyErr_NewException("pygpu.buffer.op_error", NULL, NULL);
  

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
