include "defs.pxi"

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
    cdef int PySlice_GetIndicesEx(slice_object slice, Py_ssize_t length,
                                  Py_ssize_t *start, Py_ssize_t *stop,
                                  Py_ssize_t *step,
                                  Py_ssize_t *slicelength) except -1

cdef extern from "compyte_util.h":
    size_t compyte_get_elsize(int typecode)

cdef extern from "compyte_buffer.h":
    ctypedef struct gpudata:
        pass
    ctypedef struct gpukernel:
        pass

    ctypedef struct compyte_buffer_ops:
        void *buffer_init(int devno, int *ret)
        char *buffer_error()

    compyte_buffer_ops cuda_ops
    compyte_buffer_ops opencl_ops

    ctypedef struct _GpuArray "GpuArray":
        gpudata *data
        compyte_buffer_ops *ops
        size_t *dimensions
        ssize_t *strides
        unsigned int nd
        int flags
        int typecode

    ctypedef struct _GpuKernel "GpuKernel":
        gpukernel *k
        compyte_buffer_ops *ops

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

    cdef enum ga_error:
        GA_NO_ERROR, GA_MEMORY_ERROR, GA_VALUE_ERROR, GA_IMPL_ERROR,
        GA_INVALID_ERROR, GA_UNSUPPORTED_ERROR, GA_SYS_ERROR, GA_RUN_ERROR

    cdef enum ga_usefl:
        GA_USE_CLUDA, GA_USE_SMALL, GA_USE_DOUBLE, GA_USE_COMPLEX, GA_USE_HALF

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
        GA_LONGLONG,
        GA_ULONGLONG,
        GA_FLOAT,
        GA_DOUBLE,
        GA_LONGDOUBLE,
        GA_CFLOAT,
        GA_CDOUBLE,
        GA_CLONGDOUBLE,
        GA_NBASE

    char *Gpu_error(compyte_buffer_ops *o, int err) nogil

    int GpuArray_empty(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord) nogil
    int GpuArray_fromdata(_GpuArray *a, compyte_buffer_ops *ops, gpudata *data,
                          int typecode, unsigned int nd, size_t *dims,
                          ssize_t *strides, int writable) nogil
    int GpuArray_view(_GpuArray *v, _GpuArray *a) nogil
    int GpuArray_index(_GpuArray *r, _GpuArray *a, ssize_t *starts,
                       ssize_t *stops, ssize_t *steps) nogil

    void GpuArray_clear(_GpuArray *a) nogil

    int GpuArray_share(_GpuArray *a, _GpuArray *b) nogil

    int GpuArray_move(_GpuArray *dst, _GpuArray *src) nogil
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz) nogil
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src) nogil
    int GpuArray_memset(_GpuArray *a, int data) nogil

    char *GpuArray_error(_GpuArray *a, int err) nogil

    void GpuArray_fprintf(libc.stdio.FILE *fd, _GpuArray *a) nogil
    int GpuArray_is_c_contiguous(_GpuArray *a) nogil
    int GpuArray_is_f_contiguous(_GpuArray *a) nogil

    int GpuKernel_init(_GpuKernel *k, compyte_buffer_ops *ops, void *ctx,
                       unsigned int count, char **strs, size_t *lens,
                       char *name, int flags) nogil
    void GpuKernel_clear(_GpuKernel *k) nogil
    int GpuKernel_setarg(_GpuKernel *k, unsigned int index, int typecode,
                         void *arg) nogil
    int GpuKernel_setbufarg(_GpuKernel *k, unsigned int index,
                            _GpuArray *a) nogil
    int GpuKernel_call(_GpuKernel *, size_t n) nogil
    void *(*cuda_call_compiler)(const_char_p src, size_t sz, int *ret)


IF WITH_CUDA:
    cdef object call_compiler_fn = None
    cdef void *(*call_compiler_default)(const_char_p src, size_t sz, int *ret)
    call_compiler_default = cuda_call_compiler

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

    def set_compiler_fn(fn):
        if callable(fn):
            call_compiler_fn = fn
            cuda_call_compiler = call_compiler_python
        elif fn is None:
            cuda_call_compiler = call_compiler_default
        else:
            raise ValueError("need a callable")

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

cdef int dtype_to_typecode(dtype) except -1:
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    if isinstance(dtype, np.dtype):
        res = NP_TO_TYPE.get(dtype, None)
        if res is not None:
            return res
    raise ValueError("don't know how to convert to dtype: %s"%(dtype,))

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
    pass

cdef bint py_CHKFLAGS(GpuArray a, int flags):
    return GpuArray_CHKFLAGS(&a.ga, flags)

cdef bint py_ISONESEGMENT(GpuArray a):
    return GpuArray_ISONESEGMENT(&a.ga)

cdef array_empty(GpuArray a, compyte_buffer_ops *ops, void *ctx, int typecode,
                 unsigned int nd, size_t *dims, ga_order ord):
    cdef int err
    with nogil:
        err = GpuArray_empty(&a.ga, ops, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_fromdata(GpuArray a, compyte_buffer_ops *ops, gpudata *data,
                    int typecode, unsigned int nd, size_t *dims,
                    ssize_t *strides, int writeable):
    cdef int err
    with nogil:
        err = GpuArray_fromdata(&a.ga, ops, data, typecode, nd, dims, strides,
                                writeable)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_view(GpuArray v, GpuArray a):
    cdef int err
    with nogil:
        err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_index(GpuArray r, GpuArray a, ssize_t *starts, ssize_t *stops,
                 ssize_t *steps):
    cdef int err
    with nogil:
        err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_clear(GpuArray a):
    with nogil:
        GpuArray_clear(&a.ga)

cdef bint array_share(GpuArray a, GpuArray b):
    cdef int res
    with nogil:
        res = GpuArray_share(&a.ga, &b.ga)
    return res

cdef array_move(GpuArray a, GpuArray src):
    cdef int err
    with nogil:
        err = GpuArray_move(&a.ga, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_write(GpuArray a, void *src, size_t sz):
    cdef int err
    with nogil:
        err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef array_read(void *dst, size_t sz, GpuArray src):
    cdef int err
    with nogil:
        err = GpuArray_read(dst, sz, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&src.ga, err))

cdef array_memset(GpuArray a, int data):
    cdef int err
    with nogil:
        err = GpuArray_memset(&a.ga, data)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef kernel_init(GpuKernel k, compyte_buffer_ops *ops, void *ctx,
                 unsigned int count, const_char_pp strs, size_t *len,
                 char *name, int flags):
    cdef int err
    with nogil:
        err = GpuKernel_init(&k.k, ops, ctx, count, strs, len, name, flags)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(ops, err))

cdef kernel_clear(GpuKernel k):
    with nogil:
        GpuKernel_clear(&k.k)

cdef kernel_setarg(GpuKernel k, unsigned int index, int typecode, void *arg):
    cdef int err
    with nogil:
        err = GpuKernel_setarg(&k.k, index, typecode, arg)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(k.k.ops, err))

cdef kernel_setbufarg(GpuKernel k, unsigned int index, GpuArray a):
    cdef int err
    with nogil:
        err = GpuKernel_setbufarg(&k.k, index, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(k.k.ops, err))

cdef kernel_call(GpuKernel k, size_t n):
    cdef int err
    with nogil:
        err = GpuKernel_call(&k.k, n)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(k.k.ops, err))


cdef compyte_buffer_ops *GpuArray_ops
cdef void *GpuArray_ctx

cdef compyte_buffer_ops *get_ops(kind) except NULL:
    IF WITH_OPENCL:
        if kind == "opencl":
            return &opencl_ops
    IF WITH_CUDA:
        if kind == "cuda":
            return &cuda_ops
    raise RuntimeError("Unsupported kind: %s"%(kind,))

cdef ops_kind(compyte_buffer_ops *ops):
    IF WITH_OPENCL:
        if ops == &opencl_ops:
            return "opencl"
    IF WITH_CUDA:
        if ops == &cuda_ops:
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
            raise GpuArrayException("No device %d"%(devno,))
        else:
            raise GpuArrayException(ops.buffer_error())
    return <size_t>ctx

def zeros(shape, dtype=GA_DOUBLE, order='A', context=None, kind=None):
    res = empty(shape, dtype=dtype, order=order, context=context, kind=kind)
    array_memset(res, 0)
    return res

def empty(shape, dtype=GA_DOUBLE, order='A', context=None, kind=None):
    return GpuArray(shape, dtype=dtype, order=order, context=context,
                    kind=kind)

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

def from_gpudata(size_t data, dtype, shape, kind=None, context=None,
                 strides=None, writable=True, base=None):
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

        res = new_GpuArray(ctx)
        array_fromdata(res, ops, <gpudata *>data, typecode, nd, cdims,
                       cstrides, <int>(1 if writable else 0))
        res.base = base
    finally:
        free(cdims)
        free(cstrides)
    return res

def array(proto, dtype=None, copy=True, order=None, ndmin=0, kind=None,
          context=None):
    cdef GpuArray res
    cdef GpuArray arg
    cdef GpuArray tmp
    cdef np.ndarray a
    cdef compyte_buffer_ops *ops
    cdef void *ctx
    cdef ga_order ord

    if isinstance(proto, GpuArray):
        if kind is not None or context is not None:
            raise ValueError("cannot copy GpuArray to a different context")

        arg = proto
        if not copy and \
                (dtype is None or np.dtype(dtype) == arg.dtype) and \
                (arg.ga.nd >= ndmin) and \
                (order is None or order == 'A' or
                 (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
                 (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS))):
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
        res = GpuArray(shp, dtype=(dtype or arg.dtype), order=order,
                       context=ctx_object(arg.ctx), kind=ops_kind(arg.ga.ops))
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

    res = new_GpuArray(ctx)
    array_empty(res, ops, ctx, dtype_to_typecode(a.dtype),
                np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a), ord)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
    return res

cdef new_GpuArray(void *ctx):
    cdef GpuArray res = GpuArray.__new__(GpuArray)
    res.ctx = ctx
    return res

from ..array import get_common_dtype, get_np_obj
from ..tools import ArrayArg, ScalarArg, as_argument
from ..dtypes import dtype_to_ctype

def elemwise1(a, op, oper=None, op_tmpl="res[i] = %(op)sa[i]"):
    from ..elemwise import ElemwiseKernel
    cdef GpuArray ary = a

    a_arg = as_argument(a, 'a')

    args = [ArrayArg(a.dtype, 'res'), a_arg]

    res = ary._empty_like_me()

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = ElemwiseKernel(ary.kind, ary.context, args, oper)
    k(res, a)
    return res

def ielemwise1(a, op, oper=None, op_tmpl="a[i] = %(op)sa[i]"):
    from ..elemwise import ElemwiseKernel
    cdef GpuArray ary = a

    a_arg = as_argument(a, 'a')

    args = [a_arg]

    if oper is None:
        oper = op_tmpl % {'op': op}

    k = ElemwiseKernel(ary.kind, ary.context, args, oper)
    k(a)
    return a

def elemwise2(a, op, b, out_dtype=None, oper=None,
              op_tmpl="res[i] = (%(out_t)s)%(a)s %(op)s (%(out_t)s)%(b)s"):
    from ..elemwise import ElemwiseKernel
    cdef GpuArray ary
    if isinstance(a, GpuArray):
        ary = a
        if not isinstance(b, GpuArray):
            b = numpy.asarray(b)
    elif isinstance(b, GpuArray):
        ary = b
        a = numpy.asarray(a)
    # ary will always be a or b since one of them is always a GpuArray
    if out_dtype is None:
        odtype = get_common_dtype(a, b, True)
    else:
        odtype = out_dtype

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [ArrayArg(odtype, 'res'), a_arg, b_arg]

    res = ary._empty_like_me(dtype=odtype)

    if oper is None:
        oper = op_tmpl % {'a': a_arg.expr(), 'op': op, 'b': b_arg.expr(),
                          'out_t': dtype_to_ctype(odtype)}

    k = ElemwiseKernel(ary.kind, ary.context, args, oper)
    k(res, a, b)
    return res

def ielemwise2(a, op, b, oper=None,
               op_tmpl="a[i] = a[i] %(op)s %(b)s"):
    from ..elemwise import ElemwiseKernel
    if not isinstance(b, GpuArray):
        b = numpy.asarray(b)

    a_arg = as_argument(a, 'a')
    b_arg = as_argument(b, 'b')

    args = [a_arg, b_arg]

    if oper is None:
        oper = op_tmpl % {'op': op, 'b': b_arg.expr()}

    k = ElemwiseKernel(a.kind, a.context, args, oper)
    k(a, b)
    return a

cdef class GpuArray:
    cdef _GpuArray ga
    cdef void *ctx
    cdef readonly object base

    def __dealloc__(self):
        array_clear(self)

    def __init__(self, shape, dtype=GA_DOUBLE, order='A', kind=None,
                 context=None):
        cdef size_t *cdims
        cdef compyte_buffer_ops *ops

        if kind is None:
            ops = GpuArray_ops
        else:
            ops = get_ops(kind)

        if context is None:
            self.ctx = GpuArray_ctx
        else:
            self.ctx = get_ctx(context)

        cdims = <size_t *>calloc(len(shape), sizeof(size_t))
        if cdims == NULL:
            raise MemoryError("could not allocate cdims")
        try:
            for i, d in enumerate(shape):
                cdims[i] = d
            array_empty(self, ops, self.ctx, dtype_to_typecode(dtype),
                        <unsigned int>len(shape), cdims, to_ga_order(order))
        finally:
            free(cdims)

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
        if dtype is None:
            typecode = self.ga.typecode
        else:
            typecode = dtype_to_typecode(dtype)
        res = new_GpuArray(self.ctx)
        array_empty(res, self.ga.ops, self.ctx, typecode,
                    self.ga.nd, self.ga.dimensions, to_ga_order(order))
        return res

    def copy(self, order='C'):
        cdef GpuArray res = new_GpuArray(self.ctx)
        cdef ga_order ord = to_ga_order(order)
        # XXX: support numpy order='K'
        # (which means: exactly the same layout as the source)

        if ord == GA_ANY_ORDER:
            if py_CHKFLAGS(self, GA_F_CONTIGUOUS) and \
                    not py_CHKFLAGS(self, GA_C_CONTIGUOUS):
                ord = GA_F_ORDER
            else:
                ord = GA_C_ORDER

        array_empty(res, self.ga.ops, self.ctx, self.ga.typecode,
                    self.ga.nd, self.ga.dimensions, ord)
        array_move(res, self)
        return res

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            return self.copy()

    def view(self):
        cdef GpuArray res = new_GpuArray(self.ctx)
        array_view(res, self)
        base = self
        while base.base is not None:
            base = base.base
        res.base = base
        return res

    def astype(self, dtype, order='A', copy=True):
        cdef GpuArray res
        cdef int typecode = dtype_to_typecode(dtype)
        cdef ga_order ord = to_ga_order(order)

        if ord == GA_ANY_ORDER:
            if py_CHKFLAGS(self, GA_F_CONTIGUOUS) and \
                    not py_CHKFLAGS(self, GA_C_CONTIGUOUS):
                ord = GA_F_ORDER
            else:
                ord = GA_C_ORDER

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
             (py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
            return self

        res = new_GpuArray(self.ctx)
        array_empty(res, self.ga.ops, self.ctx, typecode,
                    self.ga.nd, self.ga.dimensions, ord)
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

            res = new_GpuArray(self.ctx)
            array_index(res, self, starts, stops, steps)
        finally:
            free(starts)
            free(stops)
            free(steps)
        return res

    def __add__(self, other):
        return elemwise2(self, '+', other)

    def __iadd__(self, other):
        return ielemwise2(self, '+', other)

    def __sub__(self, other):
        return elemwise2(self, '-', other)

    def __isub__(self, other):
        return ielemwise2(self, '-', other)

    def __mul__(self, other):
        return elemwise2(self, '*', other)

    def __imul__(self, other):
        return ielemwise2(self, '*', other)

    def __div__(self, other):
        return elemwise2(self, '/', other)

    def __idiv__(self, other):
        return ielemwise2(self, '/', other)

    def __truediv__(self, other):
        np1 = get_np_obj(self)
        np2 = get_np_obj(other)
        res = (np1.__truediv__(np2)).dtype
        return elemwise2(self, '/', other, out_dtype=res)

    def __itruediv__(self, other):
        np2 = get_np_obj(other)
        kw = {}
        if self.dtype == numpy.float32 or np2.dtype == numpy.float32:
            kw['op_tmpl'] = "a[i] = (float)a[i] / (float)%(b)s"
        if self.dtype == numpy.float64 or np2.dtype == numpy.float64:
            kw['op_tmpl'] = "a[i] = (double)a[i] / (double)%(b)s"
        return ielemwise2(self, '/', other, **kw)

    def __floordiv__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {}
        if out_dtype == numpy.float32:
            kw['op_tmpl'] = "res[i] = floorf((float)%(a)s / (float)%(b)s)"
        if out_dtype == numpy.float64:
            kw['op_tmpl'] = "res[i] = floor((double)%(a)s / (double)%(b)s)"

        return elemwise2(self, '/', other, out_dtype=out_dtype, **kw)

    def __ifloordiv__(self, other):
        out_dtype = self.dtype
        kw = {}
        if out_dtype == numpy.float32:
            kw['op_tmpl'] = "a[i] = floorf((float)a[i] / (float)%(b)s)"
        if out_dtype == numpy.float64:
            kw['op_tmpl'] = "a[i] = floor((double)a[i] / (double)%(b)s)"
        return ielemwise2(self, '/', other, **kw)

    def __mod__(self, other):
        out_dtype = get_common_dtype(self, other, True)
        kw = {}
        if out_dtype == numpy.float32:
            kw['op_tmpl'] = "res[i] = fmodf((float)%(a)s, (float)%(b)s)"
        if out_dtype == numpy.float64:
            kw['op_tmpl'] = "res[i] = fmod((double)%(a)s, (double)%(b)s)"
        return elemwise2(self, '%', other, out_dtype=out_dtype, **kw)

    def __imod__(self, other):
        out_dtype = get_common_dtype(self, other, self.dtype == numpy.float64)
        kw = {}
        if out_dtype == numpy.float32:
            kw['op_tmpl'] = "a[i] = fmodf((float)a[i], (float)%(b)s)"
        if out_dtype == numpy.float64:
            kw['op_tmpl'] = "a[i] = fmod((double)a[i], (double)%(b)s)"
        return ielemwise2(self, '%', other, **kw)

    def __divmod__(a, b):
        from ..elemwise import ElemwiseKernel
        cdef GpuArray ary
        if isinstance(a, GpuArray):
            ary = a
            if not isinstance(b, GpuArray):
                b = numpy.asarray(b)
        elif isinstance(b, GpuArray):
            ary = b
            a = numpy.asarray(a)
        # ary always be a or b since one of them is always a GpuArray
        odtype = get_common_dtype(a, b, True)

        a_arg = as_argument(a, 'a')
        b_arg = as_argument(b, 'b')
        args = [ArrayArg(odtype, 'div'), ArrayArg(odtype, 'mod'), a_arg, b_arg]

        div = ary._empty_like_me(dtype=odtype)
        mod = ary._empty_like_me(dtype=odtype)

        divpart = "div[i] = (%(out_t)s)%(a)s / (%(out_t)s)%(b)s"
        modpart = "mod[i] = %(a)s %% %(b)s"
        if odtype == numpy.float32:
            divpart = "div[i] = floorf((float)%(a)s / (float)%(b)s)"
            modpart = "mod[i] = fmodf((float)%(a)s, (float)%(b)s)"
        if odtype == numpy.float64:
            divpart = "div[i] = floor((double)%(a)s / (double)%(b)s)"
            modpart = "mod[i] = fmod((double)%(a)s, (double)%(b)s)"
        tmpl = divpart+","+modpart
        ksrc = tmpl % {'a': a_arg.expr(), 'b': b_arg.expr(),
                       'out_t': dtype_to_ctype(odtype)}

        k = ElemwiseKernel(ary.kind, ary.context, args, ksrc)
        k(div, mod, a, b)
        return (div, mod)

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
            return ctx_object(self.ctx)

cdef class GpuKernel:
    cdef _GpuKernel k
    cdef void *ctx

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
            self.ctx = GpuArray_ctx
        else:
            self.ctx = get_ctx(context)

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
        kernel_init(self, ops, self.ctx, 1, s, &l, name, flags)

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
