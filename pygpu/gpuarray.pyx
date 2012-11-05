cimport libc.stdio
from libc.stdlib cimport malloc, calloc, free

# This is used in a hack to silence some over-eager warnings.
cdef extern from *:
    ctypedef object slice_object "PySliceObject *"
    ctypedef char **const_char_pp "const char **"
    ctypedef char *const_char_p "const char *"

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, size_t n)

cimport numpy as np

from cpython cimport Py_INCREF, PyNumber_Index

np.import_array()

cdef extern from "numpy/arrayobject.h":
    object _PyArray_Empty "PyArray_Empty" (int, np.npy_intp *, np.dtype, int)

# Numpy API steals dtype references and this breaks cython
cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d):
    Py_INCREF(c)
    return _PyArray_Empty(a, b, c, d)

cdef extern from "Python.h":
    int PySlice_GetIndicesEx(slice_object slice, Py_ssize_t length,
                             Py_ssize_t *start, Py_ssize_t *stop,
                             Py_ssize_t *step,
                             Py_ssize_t *slicelength) except -1

cdef extern from "compyte/types.h":
    ctypedef struct compyte_type:
        const_char_p cluda_name
        size_t size
        size_t align
        int typecode

    enum COMPYTE_TYPES:
        GA_BOOL,
        GA_BYTE,
        GA_UBYTE,
        GA_SHORT,
        GA_USHORT,
        GA_INT,
        GA_UINT,
        GA_LONG,
        GA_ULONG,
        GA_FLOAT,
        GA_DOUBLE,
        GA_CFLOAT,
        GA_CDOUBLE,
        GA_NBASE

cdef extern from "compyte/util.h":
    int compyte_register_type(compyte_type *t, int *ret)
    size_t compyte_get_elsize(int typecode)
    compyte_type *compyte_get_type(int typecode)

cdef extern from "compyte/error.h":
    cdef enum ga_error:
        GA_NO_ERROR, GA_MEMORY_ERROR, GA_VALUE_ERROR, GA_IMPL_ERROR,
        GA_INVALID_ERROR, GA_UNSUPPORTED_ERROR, GA_SYS_ERROR, GA_RUN_ERROR

cdef extern from "compyte/buffer.h":
    ctypedef struct gpudata:
        pass
    ctypedef struct gpukernel:
        pass

    ctypedef struct compyte_buffer_ops:
        void *buffer_init(int devno, int *ret)
        char *buffer_error(void *ctx)

    cdef enum ga_usefl:
        GA_USE_CLUDA, GA_USE_SMALL, GA_USE_DOUBLE, GA_USE_COMPLEX, GA_USE_HALF

    char *Gpu_error(compyte_buffer_ops *o, void *ctx, int err)
    compyte_buffer_ops *compyte_get_ops(const_char_p) nogil

cdef extern from "compyte/kernel.h":
    ctypedef struct _GpuKernel "GpuKernel":
        gpukernel *k
        compyte_buffer_ops *ops

    int GpuKernel_init(_GpuKernel *k, compyte_buffer_ops *ops, void *ctx,
                       unsigned int count, char **strs, size_t *lens,
                       char *name, int flags)
    void GpuKernel_clear(_GpuKernel *k)
    void *GpuKernel_context(_GpuKernel *k)
    int GpuKernel_setarg(_GpuKernel *k, unsigned int index, int typecode,
                         void *arg)
    int GpuKernel_setbufarg(_GpuKernel *k, unsigned int index,
                            _GpuArray *a)
    int GpuKernel_call(_GpuKernel *, size_t n)

cdef extern from "compyte/array.h":
    ctypedef struct _GpuArray "GpuArray":
        gpudata *data
        compyte_buffer_ops *ops
        size_t offset
        size_t *dimensions
        ssize_t *strides
        unsigned int nd
        int flags
        int typecode


    cdef int GA_C_CONTIGUOUS
    cdef int GA_F_CONTIGUOUS
    cdef int GA_OWNDATA
    cdef int GA_ENSURECOPY
    cdef int GA_ALIGNED
    cdef int GA_WRITEABLE
    cdef int GA_BEHAVED
    cdef int GA_CARRAY
    cdef int GA_FARRAY

    bint GpuArray_CHKFLAGS(_GpuArray *a, int fl)
    bint GpuArray_ISONESEGMENT(_GpuArray *a)

    ctypedef enum ga_order:
        GA_ANY_ORDER, GA_C_ORDER, GA_F_ORDER

    int GpuArray_empty(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)
    int GpuArray_fromdata(_GpuArray *a, compyte_buffer_ops *ops, gpudata *data,
                          size_t offset, int typecode, unsigned int nd, size_t *dims,
                          ssize_t *strides, int writable)
    int GpuArray_view(_GpuArray *v, _GpuArray *a)
    int GpuArray_index(_GpuArray *r, _GpuArray *a, ssize_t *starts,
                       ssize_t *stops, ssize_t *steps)

    void GpuArray_clear(_GpuArray *a)

    int GpuArray_share(_GpuArray *a, _GpuArray *b)
    void *GpuArray_context(_GpuArray *a)

    int GpuArray_move(_GpuArray *dst, _GpuArray *src)
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz)
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src)
    int GpuArray_memset(_GpuArray *a, int data)

    char *GpuArray_error(_GpuArray *a, int err)

    void GpuArray_fprintf(libc.stdio.FILE *fd, _GpuArray *a)
    int GpuArray_is_c_contiguous(_GpuArray *a)
    int GpuArray_is_f_contiguous(_GpuArray *a)

cdef extern from "compyte/extension.h":
    void *compyte_get_extension(const_char_p) nogil

cdef object call_compiler_fn = None

cdef void *call_compiler_python(const_char_p src, size_t sz,
                                int *ret) with gil:
    cdef bytes res
    cdef void *buf
    cdef char *tmp
    try:
        res = call_compiler_fn(src[:sz])
        buf = malloc(len(res))
        if buf == NULL:
            if ret != NULL:
                ret[0] = GA_SYS_ERROR
            return NULL
        tmp = res
        memcpy(buf, tmp, len(res))
        return buf
    except:
        # XXX: maybe should store the exception somewhere
        if ret != NULL:
            ret[0] = GA_RUN_ERROR
        return NULL

ctypedef void *(*comp_f)(const_char_p, size_t, int*)

def set_cuda_compiler_fn(fn):
    cdef void (*set_comp)(comp_f f)
    set_comp = <void (*)(comp_f)>compyte_get_extension("cuda_set_compiler")
    if set_comp == NULL:
        raise RuntimeError("cannot set compiler, extension is absent")
    if callable(fn):
        call_compiler_fn = fn
        set_comp(call_compiler_python)
    elif fn is None:
        set_comp(NULL)
    else:
        raise ValueError("needs a callable")

import numpy

cdef dict NP_TO_TYPE = {
    np.dtype('bool'): GA_BOOL,
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
}

cdef dict TYPE_TO_NP = dict((v, k) for k, v in NP_TO_TYPE.iteritems())

def register_dtype(np.dtype dtype, cname):
    cdef compyte_type *t
    cdef int typecode
    cdef char *tmp

    t = <compyte_type *>malloc(sizeof(compyte_type))
    if t == NULL:
        raise MemoryError("Can't allocate new type")
    tmp = <char *>malloc(len(cname)+1)
    if tmp == NULL:
        free(t)
        raise MemoryError
    memcpy(tmp, <char *>cname, len(cname)+1)
    t.size = dtype.itemsize
    t.align = dtype.alignment
    t.cluda_name = tmp
    typecode = compyte_register_type(t, NULL)
    if typecode == -1:
        free(tmp)
        free(t)
        raise RuntimeError("Could not register type")
    NP_TO_TYPE[dtype] = typecode
    TYPE_TO_NP[typecode] = dtype

cpdef int dtype_to_typecode(dtype) except -1:
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        res = NP_TO_TYPE.get(dtype, None)
        if res is not None:
            return res
    raise ValueError("don't know how to convert to dtype: %s"%(dtype,))

def dtype_to_ctype(dtype):
    cdef int typecode = dtype_to_typecode(dtype)
    cdef compyte_type *t = compyte_get_type(typecode)
    if t.cluda_name == NULL:
        raise ValueError("No mapping for %s"%(dtype,))
    return t.cluda_name

cdef ga_order to_ga_order(ord) except <ga_order>-2:
    if ord == "C" or ord == "c":
        return GA_C_ORDER
    elif ord == "A" or ord == "a":
        return GA_ANY_ORDER
    elif ord == "F" or ord == "f":
        return GA_F_ORDER
    else:
        raise ValueError("Valid orders are: 'A' (any), 'C' (C), 'F' (Fortran)")

class GpuArrayException(Exception):
    def __init__(self, msg, errcode):
        Exception.__init__(self, msg)
        self.errcode = errcode

cdef bint py_CHKFLAGS(GpuArray a, int flags):
    return GpuArray_CHKFLAGS(&a.ga, flags)

cdef bint py_ISONESEGMENT(GpuArray a):
    return GpuArray_ISONESEGMENT(&a.ga)

cdef array_empty(GpuArray a, compyte_buffer_ops *ops, void *ctx, int typecode,
                 unsigned int nd, size_t *dims, ga_order ord):
    cdef int err
    err = GpuArray_empty(&a.ga, ops, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_fromdata(GpuArray a, compyte_buffer_ops *ops, gpudata *data, size_t offset,
                    int typecode, unsigned int nd, size_t *dims,
                    ssize_t *strides, int writeable):
    cdef int err
    err = GpuArray_fromdata(&a.ga, ops, data, offset, typecode, nd, dims,
                            strides, writeable)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_view(GpuArray v, GpuArray a):
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_index(GpuArray r, GpuArray a, ssize_t *starts, ssize_t *stops,
                 ssize_t *steps):
    cdef int err
    err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_clear(GpuArray a):
    GpuArray_clear(&a.ga)

cdef bint array_share(GpuArray a, GpuArray b):
    return GpuArray_share(&a.ga, &b.ga)

cdef void *array_context(GpuArray a):
    return GpuArray_context(&a.ga)

cdef array_move(GpuArray a, GpuArray src):
    cdef int err
    err = GpuArray_move(&a.ga, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_write(GpuArray a, void *src, size_t sz):
    cdef int err
    err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_read(void *dst, size_t sz, GpuArray src):
    cdef int err
    err = GpuArray_read(dst, sz, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&src.ga, err), err)

cdef array_memset(GpuArray a, int data):
    cdef int err
    err = GpuArray_memset(&a.ga, data)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef const_char_p kernel_error(GpuKernel k, int err):
    return Gpu_error(k.k.ops, kernel_context(k), err)

cdef kernel_init(GpuKernel k, compyte_buffer_ops *ops, void *ctx,
                 unsigned int count, const_char_pp strs, size_t *len,
                 char *name, int flags):
    cdef int err
    err = GpuKernel_init(&k.k, ops, ctx, count, strs, len, name, flags)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(ops, kernel_context(k), err), err)

cdef kernel_clear(GpuKernel k):
    GpuKernel_clear(&k.k)

cdef void *kernel_context(GpuKernel k):
    return GpuKernel_context(&k.k)

cdef kernel_setarg(GpuKernel k, unsigned int index, int typecode, void *arg):
    cdef int err
    err = GpuKernel_setarg(&k.k, index, typecode, arg)
    if err != GA_NO_ERROR:
        raise GpuArrayException(kernel_error(k, err), err)

cdef kernel_setbufarg(GpuKernel k, unsigned int index, GpuArray a):
    cdef int err
    err = GpuKernel_setbufarg(&k.k, index, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(kernel_error(k, err), err)

cdef kernel_call(GpuKernel k, size_t n):
    cdef int err
    err = GpuKernel_call(&k.k, n)
    if err != GA_NO_ERROR:
        raise GpuArrayException(kernel_error(k, err), err)

cdef compyte_buffer_ops *GpuArray_ops
cdef void *GpuArray_ctx

cdef compyte_buffer_ops *get_ops(kind) except NULL:
    cdef compyte_buffer_ops *res
    res = compyte_get_ops(kind)
    if res == NULL:
        raise RuntimeError("Unsupported kind: %s"%(kind,))
    return res

cdef ops_kind(compyte_buffer_ops *ops):
    if ops == compyte_get_ops("opencl"):
        return "opencl"
    if ops == compyte_get_ops("cuda"):
        return "cuda"
    raise RuntimeError("Unknown ops vector")

cdef void *get_ctx(size_t ctx):
    return <void*>ctx

cdef size_t ctx_object(void *ctx):
    return <size_t>ctx

def set_kind_context(kind, size_t ctx):
    global GpuArray_ctx
    global GpuArray_ops
    GpuArray_ctx = <void *>ctx
    GpuArray_ops = get_ops(kind)

def init(kind, int devno):
    cdef int err = GA_NO_ERROR
    cdef void *ctx
    cdef compyte_buffer_ops *ops
    ops = get_ops(kind)
    ctx = ops.buffer_init(devno, &err)
    if (err != GA_NO_ERROR):
        if err == GA_VALUE_ERROR:
            raise GpuArrayException("No device %d"%(devno,), err)
        else:
            raise GpuArrayException(ops.buffer_error(NULL), err)
    return <size_t>ctx

def zeros(shape, dtype=GA_DOUBLE, order='A', context=None, kind=None,
          cls=None):
    res = empty(shape, dtype=dtype, order=order, context=context, kind=kind,
                cls=cls)
    array_memset(res, 0)
    return res

def empty(shape, dtype=GA_DOUBLE, order='A', context=None, kind=None,
          cls=None):
    cdef void *ctx
    cdef compyte_buffer_ops *ops
    cdef size_t *cdims
    cdef GpuArray res

    if kind is None:
        ops = GpuArray_ops
    else:
        ops = get_ops(kind)

    if context is None:
        ctx = GpuArray_ctx
    else:
        ctx = get_ctx(context)

    cdims = <size_t *>calloc(len(shape), sizeof(size_t))
    if cdims == NULL:
        raise MemoryError("could not allocate cdims")
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        res = new_GpuArray(cls)
        array_empty(res, ops, ctx, dtype_to_typecode(dtype),
                    <unsigned int>len(shape), cdims, to_ga_order(order))
    finally:
        free(cdims)
    return res

def asarray(a, dtype=None, order=None, context=None, kind=None):
    return array(a, dtype=dtype, order=order, copy=False, context=context,
                 kind=kind)

def ascontiguousarray(a, dtype=None, context=None, kind=None):
    return array(a, order='C', dtype=dtype, ndmin=1, copy=False,
                 context=context, kind=kind)

def asfortranarray(a, dtype=None, context=None, kind=None):
    return array(a, order='F', dtype=dtype, ndmin=1, copy=False,
                 context=context, kind=kind)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, kind=None, context=None,
                 strides=None, writable=True, base=None, cls=None):
    cdef GpuArray res
    cdef compyte_buffer_ops *ops
    cdef void *ctx
    cdef size_t *cdims
    cdef ssize_t *cstrides
    cdef unsigned int nd
    cdef size_t size
    cdef int typecode

    if kind is None:
        ops = GpuArray_ops
    else:
        ops = get_ops(kind)

    if context is None:
        ctx = GpuArray_ctx
    else:
        ctx = get_ctx(context)

    nd = <unsigned int>len(shape)
    if strides is not None and len(strides) != nd:
        raise ValueError("strides must be the same length as shape")

    typecode = dtype_to_typecode(dtype)

    try:
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
                cstrides[i] = s
        else:
            size = compyte_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
                strides[i] = size
                size *= cdims[i]

        res = new_GpuArray(cls)
        array_fromdata(res, ops, <gpudata *>data, offset, typecode, nd, cdims,
                       cstrides, <int>(1 if writable else 0))
        res.base = base
    finally:
        free(cdims)
        free(cstrides)
    return res

def array(proto, dtype=None, copy=True, order=None, ndmin=0, kind=None,
          context=None, cls=None):
    cdef GpuArray res
    cdef GpuArray arg
    cdef GpuArray tmp
    cdef np.ndarray a
    cdef compyte_buffer_ops *ops
    cdef void *ctx
    cdef ga_order ord

    if isinstance(proto, GpuArray):
        arg = proto

        if kind is not None and get_ops(kind) != arg.ga.ops:
            raise ValueError("cannot change the kind of an array")
        if context is not None and get_ctx(context) != array_context(arg):
            raise ValueError("cannot copy an array to a different context")

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (arg.ga.nd >= ndmin)
            and (order is None or order == 'A' or
                 (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
                 (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))
            and (cls is None or proto.__class__ is cls)):
            return arg
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
                order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
                order = 'F'
        if cls is None:
            cls = proto.__class__
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
                    kind=ops_kind(arg.ga.ops),
                    context=ctx_object(array_context(arg)))
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
            tmp = res
        array_move(tmp, arg)
        return res

    if kind is None:
        ops = GpuArray_ops
    else:
        ops = get_ops(kind)

    if context is None:
        ctx = GpuArray_ctx
    else:
        ctx = get_ctx(context)

    a = numpy.array(proto, dtype=dtype, order=order, ndmin=ndmin,
                    copy=False)

    if not np.PyArray_ISONESEGMENT(a):
        a = np.PyArray_ContiguousFromAny(a, np.PyArray_TYPE(a),
                                         np.PyArray_NDIM(a),
                                         np.PyArray_NDIM(a))

    if np.PyArray_ISFORTRAN(a) and not np.PyArray_ISCONTIGUOUS(a):
        ord = GA_F_ORDER
    else:
        ord = GA_C_ORDER

    res = new_GpuArray(cls)
    array_empty(res, ops, ctx, dtype_to_typecode(a.dtype),
                np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a), ord)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
    return res

cdef GpuArray new_GpuArray(cls):
    cdef GpuArray res
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
        res = GpuArray.__new__(cls)
    res.base = None
    return res

cdef public class GpuArray [type GpuArrayType, object GpuArrayObject]:
    cdef _GpuArray ga
    cdef readonly object base
    cdef object __weakref__

    def __dealloc__(self):
        array_clear(self)

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError("Called raw GpuArray.__init__")

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
                        ssize_t *stop, ssize_t *step):
        cdef Py_ssize_t dummy
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
                k += self.ga.dimensions[i]
            if k < 0 or k >= self.ga.dimensions[i]:
                raise IndexError("index %d out of bounds"%(i,))
            start[0] = k
            step[0] = 0
            return
        except TypeError:
            pass

        if isinstance(key, slice):
            # C compiler complains about argument 1 (key) because it's
            # declared as a PyObject.  But we know it's a slice so it's ok.
            PySlice_GetIndicesEx(<slice_object>key, self.ga.dimensions[i],
                                 start, stop, step, &dummy)
            if stop[0] < start[0] and step[0] > 0:
                stop[0] = start[0]
        elif key is Ellipsis:
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
            step[0] = 1
        else:
            raise IndexError("cannot index with: %s"%(key,))

    def __array__(self):
        cdef np.ndarray res

        if not py_ISONESEGMENT(self):
            self = self.copy()

        res = PyArray_Empty(self.ga.nd, <np.npy_intp *>self.ga.dimensions,
                            self.dtype, py_CHKFLAGS(self, GA_F_CONTIGUOUS) \
                                and not py_CHKFLAGS(self, GA_C_CONTIGUOUS))

        array_read(np.PyArray_DATA(res),
                   np.PyArray_NBYTES(res),
                   self)

        return res

    def _empty_like_me(self, dtype=None, order='C'):
        cdef int typecode
        cdef GpuArray res
        cdef ga_order ord = to_ga_order(order)

        # XXX: support numpy order='K'
        # (which means: as close as possible to the layout of the source)
        if ord == GA_ANY_ORDER:
            if py_CHKFLAGS(self, GA_F_CONTIGUOUS) and \
                    not py_CHKFLAGS(self, GA_C_CONTIGUOUS):
                ord = GA_F_ORDER
            else:
                ord = GA_C_ORDER

        if dtype is None:
            typecode = self.ga.typecode
        else:
            typecode = dtype_to_typecode(dtype)

        res = new_GpuArray(self.__class__)
        array_empty(res, self.ga.ops, array_context(self), typecode,
                    self.ga.nd, self.ga.dimensions, ord)
        return res

    cpdef copy(self, order='C'):
        cdef GpuArray res
        res = self._empty_like_me(order=order)
        array_move(res, self)
        return res

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            return self.copy()

    def view(self, cls=None):
        cdef GpuArray res = new_GpuArray(cls)
        array_view(res, self)
        base = self
        while hasattr(base, 'base') and base.base is not None:
            base = base.base
        res.base = base
        return res

    def astype(self, dtype, order='A', copy=True):
        cdef GpuArray res
        cdef int typecode = dtype_to_typecode(dtype)
        cdef ga_order ord = to_ga_order(order)

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
             (py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
            return self

        res = self._empty_like_me(dtype=typecode, order=order)
        array_move(res, self)
        return res

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
        else:
            raise TypeError("len() of unsized object")

    def __getitem__(self, key):
        cdef GpuArray res
        cdef ssize_t *starts
        cdef ssize_t *stops
        cdef ssize_t *steps
        cdef unsigned int i
        cdef unsigned int d
        cdef unsigned int el

        if key is Ellipsis:
            return self
        elif self.ga.nd == 0:
            raise IndexError("0-d arrays can't be indexed")

        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
                raise MemoryError

            d = 0

            if isinstance(key, tuple):
                if Ellipsis in key:
                    # The following code replaces the first Ellipsis
                    # found in the key by a bunch of them depending on
                    # the number of dimensions.  As example, this
                    # allows indexing on the last dimension with
                    # a[..., 1:] on any array (including 1-dim).  This
                    # is also required for numpy compat.
                    el = key.index(Ellipsis)
                    key = key[:el] + \
                        (Ellipsis,)*(self.ga.nd - (len(key) - 1)) + \
                        key[el+1:]
                if len(key) > self.ga.nd:
                    raise IndexError("too many indices")
                for i in range(0, len(key)):
                    self.__index_helper(key[i], i, &starts[i], &stops[i],
                                        &steps[i])
                d += <unsigned int>len(key)
            else:
                self.__index_helper(key, 0, starts, stops, steps)
                d += 1

            for i in range(d, self.ga.nd):
                starts[i] = 0
                stops[i] = self.ga.dimensions[i]
                steps[i] = 1

            base = self
            while hasattr(base, 'base') and base.base is not None:
                base = base.base
            res = new_GpuArray(self.__class__)
            array_index(res, self, starts, stops, steps)
            res.base = base
        finally:
            free(starts)
            free(stops)
            free(steps)
        return res

    def __hash__(self):
        raise TypeError("unhashable type '%s'"%self.__class__)

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
            raise ValueError("Thruth value of array with more than one element is ambiguous")

    property shape:
        "shape of this ndarray (tuple)"
        def __get__(self):
            cdef unsigned int i
            res = [None] * self.ga.nd
            for i in range(self.ga.nd):
                res[i] = self.ga.dimensions[i]
            return tuple(res)

        def __set__(self, newshape):
            raise NotImplementedError("TODO: call reshape")

    property size:
        "The number of elements in this object."
        def __get__(self):
            cdef size_t res = 1
            cdef unsigned int i
            for i in range(self.ga.nd):
                res *= self.ga.dimensions[i]
            return res

    property strides:
        "data pointer strides (in bytes)"
        def __get__(self):
            cdef unsigned int i
            res = [None] * self.ga.nd
            for i in range(self.ga.nd):
                res[i] = self.ga.strides[i]
            return tuple(res)

    property ndim:
        "The number of dimensions in this object"
        def __get__(self):
            return self.ga.nd

    property dtype:
        "The dtype of the element"
        def __get__(self):
            res = TYPE_TO_NP.get(self.ga.typecode, None)
            if res is not None:
                return res
            else:
                raise NotImplementedError("TODO")

    property typecode:
        "The compyte typecode for the data type of the array"
        def __get__(self):
            return self.ga.typecode

    property itemsize:
        "The size of the base element."
        def __get__(self):
            return compyte_get_elsize(self.ga.typecode)

    property flags:
        "Return the flags as a dictionary"
        def __get__(self):
            res = dict()
            res["C_CONTIGUOUS"] = py_CHKFLAGS(self, GA_C_CONTIGUOUS)
            res["F_CONTIGUOUS"] = py_CHKFLAGS(self, GA_F_CONTIGUOUS)
            res["WRITEABLE"] = py_CHKFLAGS(self, GA_WRITEABLE)
            res["ALIGNED"] = py_CHKFLAGS(self, GA_ALIGNED)
            res["UPDATEIFCOPY"] = False  # Unsupported
            res["OWNDATA"] = py_CHKFLAGS(self, GA_OWNDATA)
            return res

    property offset:
        "Return the offset into the gpudata pointer for this array."
        def __get__(self):
            return self.ga.offset

    property gpudata:
        "Return a pointer to the raw gpudata object."
        def __get__(self):
            return <size_t>self.ga.data

    property kind:
        "Return the kind string for the object backing this array."
        def __get__(self):
            return ops_kind(self.ga.ops)

    property context:
        "Return the context with which this array is associated."
        def __get__(self):
            return ctx_object(array_context(self))

cdef class GpuKernel:
    cdef _GpuKernel k

    def __dealloc__(self):
        kernel_clear(self)

    def __cinit__(self, source, name, kind=None, context=None, cluda=True,
                  have_double=False, have_small=False, have_complex=False,
                  have_half=False, *a, **kwa):
        cdef const_char_p s[1]
        cdef size_t l
        cdef compyte_buffer_ops *ops
        cdef int flags = 0

        if not isinstance(source, (str, unicode)):
            raise TypeError("Expected a string for the kernel source")
        if not isinstance(name, (str, unicode)):
            raise TypeError("Expected a string for the kernel name")

        if kind is None:
            ops = GpuArray_ops
        else:
            ops = get_ops(kind)

        if context is None:
            ctx = GpuArray_ctx
        else:
            ctx = get_ctx(context)

        if cluda:
            flags |= GA_USE_CLUDA
        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
            flags |= GA_USE_HALF

        # This is required under CUDA otherwise the function is compiled
        # as a C++ mangled name and is irretriveable
        if kind == "cuda":
            ss = 'extern "C" {%s}'%(source,)
        else:
            ss = source

        s[0] = ss
        l = len(ss)
        kernel_init(self, ops, ctx, 1, s, &l, name, flags)

    def __call__(self, *args, n=None):
        if n is None:
            raise ValueError("Must specify size (n)")
        self.setargs(args)
        self.call(n)

    cpdef setargs(self, args):
        # Work backwards to avoid a lot of reallocations in the argument code.
        for i in range(len(args)-1, -1, -1):
            self.setarg(i, args[i])

    def setarg(self, unsigned int index, o):
        if isinstance(o, GpuArray):
            self.setbufarg(index, o)
        else:
            # this will break for objects that are not numpy-like, but meh.
            self._setarg(index, o.dtype, o)

    def setbufarg(self, unsigned int index, GpuArray a not None):
        kernel_setbufarg(self, index, a)

    cdef _setarg(self, unsigned int index, np.dtype t, object o):
        cdef float f
        cdef double d
        cdef signed char b
        cdef unsigned char ub
        cdef short s
        cdef unsigned short us
        cdef int i
        cdef unsigned int ui
        cdef long l
        cdef unsigned long ul
        cdef unsigned int typecode
        typecode = dtype_to_typecode(t)
        if typecode == GA_FLOAT:
            f = o
            kernel_setarg(self, index, typecode, &f)
        elif typecode == GA_DOUBLE:
            d = o
            kernel_setarg(self, index, typecode, &d)
        elif typecode == GA_BYTE:
            b = o
            kernel_setarg(self, index, typecode, &b)
        elif typecode == GA_UBYTE:
            ub = o
            kernel_setarg(self, index, typecode, &ub)
        elif typecode == GA_SHORT:
            s = o
            kernel_setarg(self, index, typecode, &s)
        elif typecode == GA_USHORT:
            us = o
            kernel_setarg(self, index, typecode, &us)
        elif typecode == GA_INT:
            i = o
            kernel_setarg(self, index, typecode, &i)
        elif typecode == GA_UINT:
            ui = o
            kernel_setarg(self, index, typecode, &ui)
        elif typecode == GA_LONG:
            l = o
            kernel_setarg(self, index, typecode, &l)
        elif typecode == GA_ULONG:
            ul = o
            kernel_setarg(self, index, typecode, &ul)
        else:
            raise ValueError("Can't set argument of this type")

    cpdef call(self, size_t n):
        kernel_call(self, n)
