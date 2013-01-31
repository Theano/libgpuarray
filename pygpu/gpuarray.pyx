cimport libc.stdio
from libc.stdlib cimport malloc, calloc, free

# This is used in a hack to silence some over-eager warnings.
cdef extern from *:
    ctypedef object slice_object "PySliceObject *"
    ctypedef char **const_char_pp "const char **"
    ctypedef char *const_char_p "const char *"

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, size_t n)
    void *memset(void *b, int c, size_t sz)

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
        void buffer_deinit(void *ctx)
        char *buffer_error(void *ctx)
        int buffer_property(void *c, gpudata *b, gpukernel *k, int prop_id,
                            void *res)

    int GA_CTX_PROP_DEVNAME
    int GA_CTX_PROP_MAXLSIZE
    int GA_CTX_PROP_LMEMSIZE
    int GA_CTX_PROP_NUMPROCS
    int GA_BUFFER_PROP_CTX
    int GA_KERNEL_PROP_CTX
    int GA_KERNEL_PROP_MAXLSIZE
    int GA_KERNEL_PROP_PREFLSIZE
    int GA_KERNEL_PROP_MAXGSIZE

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
    int GpuKernel_call(_GpuKernel *, size_t n, size_t ls, size_t gs)

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
    int GpuArray_sync(_GpuArray *a)
    int GpuArray_index(_GpuArray *r, _GpuArray *a, ssize_t *starts,
                       ssize_t *stops, ssize_t *steps)
    int GpuArray_reshape(_GpuArray *res, _GpuArray *a, unsigned int nd,
                         size_t *newdims, ga_order ord, int nocopy)

    void GpuArray_clear(_GpuArray *a)

    int GpuArray_share(_GpuArray *a, _GpuArray *b)
    void *GpuArray_context(_GpuArray *a)

    int GpuArray_move(_GpuArray *dst, _GpuArray *src)
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz)
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src)
    int GpuArray_memset(_GpuArray *a, int data)
    int GpuArray_copy(_GpuArray *res, _GpuArray *a, ga_order order)

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
    """
    set_cuda_compiler_fn(fn)

    Sets the compiler function for cuda kernels.

    :param fn: compiler function
    :type fn: callable
    :rtype: None

    `fn` must have the following signature::

        fn(source)

    It will recieve a python bytes string consiting the of complete
    kernel source code and must return a python byte string consisting
    of the compilation results or raise an exception.

    .. warning::

        Exceptions raised by the function will not be propagated
        because the call path goes through libcompyte.  They are only
        used to indicate that there was a problem during the
        compilation.

    This overrides the built-in compiler function with the provided
    one or resets to the default if `None` is given.  The provided
    function must be rentrant if the library is used in a
    multi-threaded context.

    .. note::
        If the "cuda" module was not compiled in libcompyte then this function will raise a `RuntimeError` unconditionaly.
    """
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
    """
    register_dtype(dtype, cname)

    Make a new type known to the cluda machinery.

    This function return the associted internal typecode for the new
    type.

    :param dtype: new type
    :type dtype: numpy.dtype
    :param cname: C name for the type declarations
    :type cname: string
    :rtype: int
    """
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

cdef public np.dtype typecode_to_dtype(int typecode):
    res = TYPE_TO_NP.get(typecode, None)
    if res is not None:
        return res
    else:
        raise NotImplementedError("TODO")

cpdef int dtype_to_typecode(dtype) except -1:
    """
    dtype_to_typecode(dtype)

    Get the internal typecode for a type.

    :param dtype: type to get the code for
    :type dtype: numpy.dtype
    :rtype: int
    """
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
    """
    dtype_to_ctype(dtype)

    Return the C name for a type.

    :param dtype: type to get the name for
    :type dtype: numpy.dtype
    :rtype: string
    """
    cdef int typecode = dtype_to_typecode(dtype)
    cdef compyte_type *t = compyte_get_type(typecode)
    if t.cluda_name == NULL:
        raise ValueError("No mapping for %s"%(dtype,))
    return t.cluda_name

cdef ga_order to_ga_order(ord) except <ga_order>-2:
    if ord == "C" or ord == "c":
        return GA_C_ORDER
    elif ord == "A" or ord == "a" or ord is None:
        return GA_ANY_ORDER
    elif ord == "F" or ord == "f":
        return GA_F_ORDER
    else:
        raise ValueError("Valid orders are: 'A' (any), 'C' (C), 'F' (Fortran)")

class GpuArrayException(Exception):
    """
    Exception used for all errors related to libcompyte.
    """
    def __init__(self, msg, errcode):
        """
        __init__(self, msg, errcode)
        """
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
        raise GpuArrayException(Gpu_error(ops, ctx, err), err)

cdef array_fromdata(GpuArray a, compyte_buffer_ops *ops, gpudata *data,
                    size_t offset, int typecode, unsigned int nd, size_t *dims,
                    ssize_t *strides, int writeable):
    cdef int err
    cdef void *ctx
    err = GpuArray_fromdata(&a.ga, ops, data, offset, typecode, nd, dims,
                            strides, writeable)
    if err != GA_NO_ERROR:
        ops.buffer_property(NULL, data, NULL, GA_BUFFER_PROP_CTX, &ctx)
        raise GpuArrayException(Gpu_error(ops, ctx, err), err)

cdef array_view(GpuArray v, GpuArray a):
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_sync(GpuArray a):
    cdef int err
    err = GpuArray_sync(&a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_index(GpuArray r, GpuArray a, ssize_t *starts, ssize_t *stops,
                 ssize_t *steps):
    cdef int err
    err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err), err)

cdef array_reshape(GpuArray res, GpuArray a, unsigned int nd, size_t *newdims,
                   ga_order ord, int nocopy):
    cdef int err
    err = GpuArray_reshape(&res.ga, &a.ga, nd, newdims, ord, nocopy)
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

cdef array_copy(GpuArray res, GpuArray a, ga_order order):
    cdef int err
    err = GpuArray_copy(&res.ga, &a.ga, order)
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
        raise GpuArrayException(Gpu_error(ops, ctx, err), err)

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

cdef kernel_call(GpuKernel k, size_t n, size_t ls, size_t gs):
    cdef int err
    err = GpuKernel_call(&k.k, n, ls, gs)
    if err != GA_NO_ERROR:
        raise GpuArrayException(kernel_error(k, err), err)

cdef kernel_property(GpuKernel k, int prop_id, void *res):
    cdef int err
    err = k.k.ops.buffer_property(NULL, NULL, k.k.k, prop_id, res)
    if err != GA_NO_ERROR:
        raise GpuArrayException(kernel_error(k, err), err)

cdef GpuContext GpuArray_default_context = None

cdef ctx_property(GpuContext c, int prop_id, void *res):
    cdef int err
    err = c.ops.buffer_property(c.ctx, NULL, NULL, prop_id, res)
    if err != GA_NO_ERROR:
        raise GpuArrayException(Gpu_error(c.ops, c.ctx, err), err)

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

def set_default_context(GpuContext ctx):
    """
    set_default_context(ctx)

    Set the default context for the module.

    :param ctx: default context
    :type ctx: GpuContext
    :rtype: None

    The provided context will be used as a default value for all the
    other functions in this module which take a context as parameter.
    Call with `None` to clear the default value.

    If you don't call this function the context of all other functions
    is a mandatory argument.

    This can be helpful to reduce clutter when working with only one
    context. It is strongly discouraged to use this function when
    working with multiple contexts at once.
    """
    global GpuArray_default_context
    GpuArray_default_context = ctx

cdef GpuContext ensure_context(GpuContext c):
    if c is None:
        if GpuArray_default_context is None:
            raise TypeError("No context specified.")
        return GpuArray_default_context
    return c

def init(dev):
    """
    init(dev)

    Creates a context from a device specifier.

    :param dev: device specifier
    :type dev: string
    :rtype: GpuContext

    Device specifiers are composed of the type string and the device
    id like so::

        "cuda0"
        "opencl0:1"

    For cuda the device id is the numeric identifier.  You can see
    what devices are available by running nvidia-smi on the machine.

    For opencl the device id is the platform number, a colon (:) and
    the device number.  There are no widespread and/or easy way to
    list available platforms and devices.  You can experiement with
    the values, unavaiable ones will just raise an error, and there
    are no gaps in the valid numbers.
    """
    if dev.startswith('cuda'):
        kind = "cuda"
        devnum = int(dev[4:])
    elif dev.startswith('opencl'):
        kind = "opencl"
        devspec = dev[6:].split(':')
        devnum = int(devspec[0]) << 16 | int(devspec[1])
    else:
        raise ValueError("Unknown device format:", dev)
    return GpuContext(kind, devnum)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
    zeros(shape, dtype='float64', order='C', context=None, cls=None)

    Returns an array of zero-initialized values of the requested
    shape, type and order.

    :param shape: number of elements in each dimension
    :type shape: iterable of ints
    :param dtype: type of the elements
    :type dtype: string, numpy.dtype or int
    :param order: layout of the data in memory, one of 'A'ny, 'C' or 'F'ortran
    :type order: string
    :param context: context in which to do the allocation
    :type context: GpuContext
    :param cls: class of the returned array (must inherit from GpuArray)
    :type cls: class
    :rtype: array
    """
    res = empty(shape, dtype=dtype, order=order, context=context, cls=cls)
    array_memset(res, 0)
    return res

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
    empty(shape, dtype='float64', order='C', context=None, cls=None)

    Returns an empty (uninitialized) array of the requested shape,
    type and order.

    :param shape: number of elements in each dimension
    :type shape: iterable of ints
    :param dtype: type of the elements
    :type dtype: string, numpy.dtype or int
    :param order: layout of the data in memory, one of 'A'ny, 'C' or 'F'ortran
    :type order: string
    :param context: context in which to do the allocation
    :type context: GpuContext
    :param cls: class of the returned array (must inherit from GpuArray)
    :type cls: class
    :rtype: array
    """
    cdef size_t *cdims
    cdef GpuArray res

    context = ensure_context(context)

    cdims = <size_t *>calloc(len(shape), sizeof(size_t))
    if cdims == NULL:
        raise MemoryError("could not allocate cdims")
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        res = new_GpuArray(cls, context)
        array_empty(res, context.ops, context.ctx, dtype_to_typecode(dtype),
                    <unsigned int>len(shape), cdims, to_ga_order(order))
    finally:
        free(cdims)
    return res

def asarray(a, dtype=None, order='A', GpuContext context=None):
    """
    asarray(a, dtype=None, order='A', context=None)

    Returns a GpuArray from the data in `a`

    :param a: data
    :type shape: array-like
    :param dtype: type of the elements
    :type dtype: string, numpy.dtype or int
    :param order: layout of the data in memory, one of 'A'ny, 'C' or 'F'ortran
    :type order: string or int
    :param context: context in which to do the allocation
    :type context: GpuContext
    :rtype: GpuArray

    If `a` is already a GpuArray and all other parameters match, then
    the object itself returned.  If `a` is an instance of a subclass
    of GpuArray then a view of the base class will be returned.
    Otherwise a new object is create and the data is copied into it.

    `context` is optional if `a` is a GpuArray (but must match exactly
    the context of `a` if specified) and is mandatory otherwise.
    """
    return array(a, dtype=dtype, order=order, copy=False, context=context,
                 cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
    """
    ascontiguousarray(a, dtype=None, context=None)

    Returns a contiguous array in device memory (C order).

    :param a: input
    :type a: array-like
    :param dtype: type of the return array
    :type dtype: string, numpy.dtype or int
    :param context: context to use for a new array
    :type context: GpuContext
    :rtype: array

    `context` is optional if `a` is a GpuArray (but must match exactly
    the context of `a` if specified) and is mandatory otherwise.
    """
    return array(a, order='C', dtype=dtype, ndmin=1, copy=False,
                 context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
    """
    asfortranarray(a, dtype=None, context=None)

    Returns a contiguous array in device memory (Fortran order)

    :param a: input
    :type a: array-like
    :param dtype: type of the elements
    :type dtype: string, numpy.dtype or int
    :param context: context in which to do the allocation
    :type context: GpuContext
    :rtype: array

    `context` is optional if `a` is a GpuArray (but must match exactly
    the context of `a` if specified) and is mandatory otherwise.
    """
    return array(a, order='F', dtype=dtype, ndmin=1, copy=False,
                 context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    """
    may_share_memory(a, b)

    Returns True if `a` and `b` may share memory, False otherwise.
    """
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
                 strides=None, writable=True, base=None, cls=None):
    """
    from_gpudata(data, offset, dtype, shape, context=None, strides=None, writable=True, base=None, cls=None)

    Build a GpuArray from pre-allocated gpudata

    :param data: pointer to a gpudata structure
    :type data: int
    :param offset: offset to the data location inside the gpudata
    :type offset: int
    :param dtype: data type of the gpudata elements
    :type dtype: numpy.dtype
    :param shape: shape to use for the result
    :type shape: iterable of ints
    :param context: context of the gpudata
    :type context: GpuContext
    :param strides: strides for the results
    :type strides: iterable of ints
    :param writable: is the data writable?
    :type writeable: bool
    :param base: base object that keeps gpudata alive
    :param cls: view type of the result

    .. warning::
        This function is intended for advanced use and will crash the
        interpreter if used improperly.

    .. note::
        This function might be deprecated in a later relase since the
        only way to create gpudata pointers is through libcompyte
        functions that aren't exposed at the python level. It can be
        used with the value of the `gpudata` attribute of an existing
        GpuArray.
    """
    cdef GpuArray res
    cdef size_t *cdims
    cdef ssize_t *cstrides
    cdef unsigned int nd
    cdef size_t size
    cdef int typecode

    context = ensure_context(context)

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

        res = new_GpuArray(cls, context)
        array_fromdata(res, context.ops, <gpudata *>data, offset, typecode,
                       nd, cdims, cstrides, <int>(1 if writable else 0))
        res.base = base
    finally:
        free(cdims)
        free(cstrides)
    return res

def array(proto, dtype=None, copy=True, order=None, ndmin=0,
          GpuContext context=None, cls=None):
    """
    array(obj, dtype='float64', copy=True, order=None, ndmin=0, context=None, cls=None)

    Create a GpuArray from existing data

    :param obj: data to initialize the result
    :type obj: array-like
    :param dtype: data type of the result elements
    :type dtype: string or numpy.dtype or int
    :param copy: return a copy?
    :type copy: bool
    :param order: memory layout of the result
    :type order: string
    :param ndmin: minimum number of result dimensions
    :type ndmin: int
    :param context: allocation context
    :type context: GpuContext
    :param cls: result class (must inherit from GpuArray)
    :type cls: class
    :rtype: GpuArray

    This function creates a new GpuArray from the data provided in
    `obj` except if `obj` is already a GpuArray and all the parameters
    match its properties and `copy` is False.

    The properties of the resulting array depend on the input data
    except if overriden by other parameters.

    This function is similar to :meth:`numpy.array` except that it returns
    GpuArrays.
    """
    cdef GpuArray res
    cdef GpuArray arg
    cdef GpuArray tmp
    cdef np.ndarray a
    cdef ga_order ord

    if isinstance(proto, GpuArray):
        arg = proto

        if context is not None and  context.ctx != array_context(arg):
            raise ValueError("cannot copy an array to a different context")

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (arg.ga.nd >= ndmin)
            and (order is None or order == 'A' or
                 (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
                 (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if cls is None or arg.__class__ is cls:
                return arg
            else:
                return arg.view(cls)

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
                    context=arg.context)
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
            tmp = res
        array_move(tmp, arg)
        return res

    context = ensure_context(context)

    a = numpy.array(proto, dtype=dtype, order=order, ndmin=ndmin,
                    copy=False)

    if not np.PyArray_ISONESEGMENT(a):
        a = np.PyArray_GETCONTIGUOUS(a)

    if np.PyArray_ISFORTRAN(a) and not np.PyArray_ISCONTIGUOUS(a):
        ord = GA_F_ORDER
    else:
        ord = GA_C_ORDER

    res = new_GpuArray(cls, context)
    array_empty(res, context.ops, context.ctx, dtype_to_typecode(a.dtype),
                np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a), ord)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
    return res

cdef public class GpuContext [type GpuContextType, object GpuContextObject]:
    """
    Class that holds all the information pertaining to a context.

    .. code-block:: python

        GpuContext(kind, devno)

    :param kind: module name for the context
    :type kind: string
    :param devno: device number
    :type devno: int

    The currently implemented modules (for the `kind` parameter) are
    "cuda" and "opencl".  Which are available depends on the build
    options for libcompyte.

    If you want an alternative interface check :meth:`~pygpu.gpuarray.init`.
    """
    cdef compyte_buffer_ops *ops
    cdef void* ctx

    def __dealloc__(self):
        if self.ctx != NULL:
            self.ops.buffer_deinit(self.ctx)

    def __cinit__(self, kind, devno, *args, **kwargs):
        cdef int err = GA_NO_ERROR
        cdef void *ctx
        self.ops = get_ops(kind)
        self.ctx = self.ops.buffer_init(devno, &err)
        if (err != GA_NO_ERROR):
            if err == GA_VALUE_ERROR:
                raise GpuArrayException("No device %d"%(devno,), err)
            else:
                raise GpuArrayException(self.ops.buffer_error(NULL), err)

    property kind:
        "Module name this context uses"
        def __get__(self):
            return ops_kind(self.ops)

    property ptr:
        "Raw pointer value for the context object"
        def __get__(self):
            return <size_t>self.ctx

    property devname:
        "Device name for this context"
        def __get__(self):
            cdef char *tmp
            cdef unicode res

            ctx_property(self, GA_CTX_PROP_DEVNAME, &tmp)
            try:
                res = tmp.decode('ascii')
            finally:
                free(tmp)
            return res

    property maxlsize:
        "Maximum size of thread block (local size) for this context"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE, &res)
            return res

    property lmemsize:
        "Size of the local (shared) memory, in bytes, for this context"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LMEMSIZE, &res)
            return res

    property numprocs:
        "Number of compute units for this context"
        def __get__(self):
            cdef unsigned int res
            ctx_property(self, GA_CTX_PROP_NUMPROCS, &res)
            return res

cdef public GpuArray new_GpuArray(cls, GpuContext ctx):
    cdef GpuArray res
    if ctx is None:
        raise RuntimeError("ctx is None in new_GpuArray")
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
        res = GpuArray.__new__(cls)
    res.base = None
    res.context = ctx
    return res

cdef public class GpuArray [type GpuArrayType, object GpuArrayObject]:
    """
    Device array

    To create instances of this class use
    :meth:`~pygpu.gpuarray.zeros`, :meth:`~pygpu.gpuarray.empty` or
    :meth:`~pygpu.gpuarray.array`.  It cannot be instanciated
    directly.

    You can also subclass this class and make the module create your
    instances by passing the `cls` agument to any method that return a
    new GpuArray.  This way of creating the class will NOT call your
    :meth:`__init__` method.

    You can also implement your own :meth:`__init__` method, but you
    must take care to ensure you properly initialized the GpuArray C
    fields before using it or you will most likely crash the
    interpreter.
    """
    cdef _GpuArray ga
    cdef readonly GpuContext context
    cdef readonly object base
    cdef object __weakref__

    def __dealloc__(self):
        array_clear(self)

    def __cinit__(self):
        memset(&self.ga, 0, sizeof(_GpuArray))

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
        """
        __array__()

        Return a :class:`numpy.ndarray` with the same content.

        Automatically used by :meth:`numpy.asarray`.
        """
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
        """
        _empty_like_me(dtype=None, order='C')

        Returns an empty (uninitialized) GpuArray with the same
        properties except if overridden by parameters.
        """
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

        res = new_GpuArray(self.__class__, self.context)
        array_empty(res, self.ga.ops, array_context(self), typecode,
                    self.ga.nd, self.ga.dimensions, ord)
        return res

    cpdef copy(self, order='C'):
        """
        copy(order='C')

        Return a copy if this array.

        :param order: memory layout of the copy
        :type order: string
        """
        cdef GpuArray res
        res = new_GpuArray(self.__class__, self.context)
        array_copy(res, self, to_ga_order(order))
        return res

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            return self.copy()

    def sync(self):
        """
        sync()

        Wait for all pending operations on this array.

        This is done automatically when reading or writing from it,
        but can be useful as a separate operation for timings.
        """
        array_sync(self)

    def view(self, cls=GpuArray):
        """
        view(cls=GpuArray)

        Return a view of this array.

        :param cls: class of the view (must inherit from GpuArray)

        The returned array shares device data with this one and both
        will reflect changes made to the other.
        """
        cdef GpuArray res = new_GpuArray(cls, self.context)
        array_view(res, self)
        if self.base is not None:
            res.base = self.base
        else:
            res.base = self
        return res

    def astype(self, dtype, order='A', copy=True):
        """
        astype(dtype, order='A', copy=True)

        Cast the elements of this array to a new type.

        :param dtype: type of the elements of the result
        :type dtype: string or numpy.dtype or int
        :param order: memory layout of the result
        :type order: string
        :param copy: Always return a copy?
        :type copy: bool

        This function returns a new array will all elements cast to
        the supplied `dtype`, but otherwise unchanged.

        If `copy` is False and the type and order match `self` is
        returned.
        """
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

    def reshape(self, shape, order='C'):
        """
        reshape(shape, order='C')

        Returns a new array with the given shape and order.

        The new shape must have the same size (total number of
        elements) as the current one.
        """
        cdef size_t *newdims
        cdef unsigned int nd
        cdef unsigned int i
        cdef GpuArray res
        nd = len(shape)
        newdims = <size_t *>calloc(nd, sizeof(size_t))
        try:
            for i in range(nd):
                newdims[i] = shape[i]
            res = new_GpuArray(self.__class__, self.context)
            array_reshape(res, self, nd, newdims, to_ga_order(order), 0)
        finally:
            free(newdims)
        if not py_CHKFLAGS(res, GA_OWNDATA):
            if self.base is not None:
                res.base = self.base
            else:
                res.base = self
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

            if self.base is not None:
                base = self.base
            else:
                base = self
            res = new_GpuArray(self.__class__, self.context)
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
            cdef size_t *newdims
            cdef unsigned int nd
            cdef unsigned int i
            cdef GpuArray res
            nd = len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
                raise MemoryError("calloc")
            try:
                for i in range(nd):
                    newdims[i] = newshape[i]
                res = new_GpuArray(GpuArray, self.context)
                array_reshape(res, self, nd, newdims, GA_C_ORDER, 1)
            finally:
                free(newdims)
            # This is safe becase the reshape above is a nocopy one
            free(self.ga.dimensions)
            free(self.ga.strides)
            self.ga.dimensions = res.ga.dimensions
            self.ga.strides = res.ga.strides
            self.ga.nd = nd
            res.ga.dimensions = NULL
            res.ga.strides = NULL
            array_clear(res)

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
            return typecode_to_dtype(self.ga.typecode)

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

cdef class GpuKernel:
    """
    .. code-block:: python

        GpuKernel(source, name, context=None, cluda=True, have_double=False, have_small=False, have_complex=False, have_half=False)

    Compile a kernel on the device

    :param source: complete kernel source code
    :type source: string
    :param name: function name of the kernel
    :type name: string
    :param context: device on which the kernel is compiled
    :type context: GpuContext
    :param cluda: use cluda layer?
    :param have_double: ensure working doubles?
    :param have_small: ensure types smaller than float will work?
    :param have_complex: ensure complex types will work?
    :param have_half: ensure half-floats will work?

    The kernel function is retrieved using the provided `name` which
    must match what you named your kernel in `source`.  You can safely
    reuse the same name multiple times.

    .. note::

        With the cuda backend, unless you use `cluda=True`, you must
        either pass the mangled name of your kernel or declare the
        function 'extern "C"', because cuda uses a C++ compiler
        unconditionally.

    The `have_*` parameter are there to tell libcompyte that we need
    the particular type or feature to work for this kernel.  If the
    request can't be satified a
    :class:`~pygpu.gpuarray.GpuArrayException` will be raised in the
    constructor.

    .. warning::

        If you do not set the `have_` flags properly, you will either
        get a device-specific error (the good case) or silent
        completly bogus data (the bad case).

    Once you have the kernel object you can simply call it like so::

        k = GpuKernel(...)
        k(param1, param2, n=n)

    where `n` is the minimum number of threads to run.  libcompyte
    will try to stay close to this number but may run a few more
    threads to match the hardware preferred multiple and stay
    efficient.  You should watch out for this in your code and make
    sure to test against the size of your data.

    If you want more control over thread allocation you can use the
    `ls` and `gs` parameters like so::

        k = GpuKernel(...)
        k(param1, param2, ls=ls, gs=gs)

    If you choose to use this interface, make sure to stay within the
    limits of `k.maxlsize` and `k.maxgsize` or the call will fail.
    """
    cdef _GpuKernel k

    def __dealloc__(self):
        kernel_clear(self)

    def __cinit__(self, source, name, GpuContext context=None, cluda=True,
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

        context = ensure_context(context)

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

        s[0] = source
        l = len(source)
        kernel_init(self, context.ops, context.ctx, 1, s, &l, name, flags)

    def __call__(self, *args, n=0, ls=0, gs=0):
        if n == 0 and (ls == 0 or gs == 0):
            raise ValueError("Must specify size (n) or both gs and ls")
        self.setargs(args)
        self.call(n, ls, gs)

    cpdef setargs(self, args):
        """
        setargs(args)

        Sets the arguments of the kernel to prepare for a :meth:`.call`

        :param args: kernel arguments
        :type args: tuple or list
        """
        # Work backwards to avoid a lot of reallocations in the argument code.
        for i in range(len(args)-1, -1, -1):
            self.setarg(i, args[i])

    def setarg(self, unsigned int index, o):
        """
        setarg(index, o)

        Set argument `index` to `o`.

        :param index: argument index
        :type index: int
        :param o: argument value
        :type o: GpuArray or numpy.ndarray

        This overwrites any previous argument set for that index.

        The type of scalar arguments is indicated by wrapping them in
        a numpy.ndarray like this::

            param1 = numpy.asarray(1.0, dtype='float32')

        Arguments which are not wrapped will raise an exception.
        """
        if isinstance(o, GpuArray):
            kernel_setbufarg(self, index, o)
            # This is to keep the reference alive
        else:
            try:
                self._setarg(index, o.dtype, o)
            except AttributeError:
                raise TypeError("Wrap your scalar arguments in numpy objects")

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
            raise TypeError("Can't set argument of this type", t)

    cpdef call(self, size_t n, size_t ls, size_t gs):
        """
        call(n, ls, gs)

        Call the kernel with the prepered arguments

        :param n: number of work elements
        :param ls: local size
        :param gs: global size

        Either `n` or `gs` and `ls` must be set (not 0).  You can also
        set `n` and one of `ls` or `gs` and the other will be filled
        in.

        For a friendlier interface try just calling the object
        (documentation is in the class doc :class:`.GpuKernel`)
        """
        kernel_call(self, n, ls, gs)

    property maxlsize:
        "Maximum local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXLSIZE, &res)
            return res

    property preflsize:
        "Preferred multiple for local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_PREFLSIZE, &res)
            return res

    property maxgsize:
        "Maximum global size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXGSIZE, &res)
            return res
