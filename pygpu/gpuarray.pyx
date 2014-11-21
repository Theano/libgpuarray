cimport libc.stdio
from libc.stdlib cimport malloc, calloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport strncmp

cimport numpy as np

from cpython cimport Py_INCREF, PyNumber_Index
from cpython.object cimport Py_EQ, Py_NE

def api_version():
    # major, minor
    return (1, 0)

np.import_array()

# to export the numeric value
SIZE = GA_SIZE

# Numpy API steals dtype references and this breaks cython
cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d):
    Py_INCREF(c)
    return _PyArray_Empty(a, b, c, d)

cdef object call_compiler_fn = None

cdef void *call_compiler_python(const char *src, size_t sz,
                                size_t *bin_len, int *ret) with gil:
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
        bin_len[0] = len(res);
        return buf
    except:
        # XXX: maybe should store the exception somewhere
        if ret != NULL:
            ret[0] = GA_RUN_ERROR
        return NULL

ctypedef void *(*comp_f)(const char *, size_t, size_t *, int*)

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
        because the call path goes through libgpuarray.  They are only
        used to indicate that there was a problem during the
        compilation.

    This overrides the built-in compiler function with the provided
    one or resets to the default if `None` is given.  The provided
    function must be rentrant if the library is used in a
    multi-threaded context.

    .. note::
        If the "cuda" module was not compiled in libgpuarray then this function will raise a `RuntimeError` unconditionaly.
    """
    cdef void (*set_comp)(comp_f f)
    set_comp = <void (*)(comp_f)>gpuarray_get_extension("cuda_set_compiler")
    if set_comp == NULL:
        raise RuntimeError, "cannot set compiler, extension is absent"
    if callable(fn):
        call_compiler_fn = fn
        set_comp(call_compiler_python)
    elif fn is None:
        set_comp(NULL)
    else:
        raise ValueError, "needs a callable"

def cl_wrap_ctx(size_t ptr):
    """
    cl_wrap_ctx(ptr)

    Wrap an existing OpenCL context (the cl_context struct) into a
    GpuContext class.
    """
    cdef void *(*cl_make_ctx)(void *)
    cdef GpuContext res
    cl_make_ctx = <void *(*)(void *)>gpuarray_get_extension("cl_make_ctx")
    if cl_make_ctx == NULL:
        raise RuntimeError, "cl_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
    res.ops = get_ops('opencl')
    res.ctx = cl_make_ctx(<void *>ptr)
    if res.ctx == NULL:
        raise RuntimeError, "cl_make_ctx call failed"
    return res

def cuda_wrap_ctx(size_t ptr, bint own):
    """
    cuda_wrap_ctx(ptr)

    Wrap an existing CUDA driver context (CUcontext) into a GpuContext
    class.

    If `own` is true, libgpuarray is now reponsible for the context and
    it will be destroyed once there are no references to it.
    Otherwise, the context will not be destroyed and it is the calling
    code's reponsability.
    """
    cdef void *(*cuda_make_ctx)(void *, int)
    cdef int flags
    cdef GpuContext res
    cuda_make_ctx = <void *(*)(void *, int)>gpuarray_get_extension("cuda_make_ctx")
    if cuda_make_ctx == NULL:
        raise RuntimeError, "cuda_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
    res.ops = get_ops('cuda')
    flags = 0
    if not own:
        flags |= GPUARRAY_CUDA_CTX_NOFREE
    res.ctx = cuda_make_ctx(<void *>ptr, flags)
    if res.ctx == NULL:
        raise RuntimeError, "cuda_make_ctx call failed"
    return res

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
    cdef gpuarray_type *t
    cdef int typecode
    cdef char *tmp

    t = <gpuarray_type *>malloc(sizeof(gpuarray_type))
    if t == NULL:
        raise MemoryError, "Can't allocate new type"
    tmp = <char *>malloc(len(cname)+1)
    if tmp == NULL:
        free(t)
        raise MemoryError
    memcpy(tmp, <char *>cname, len(cname)+1)
    t.size = dtype.itemsize
    t.align = dtype.alignment
    t.cluda_name = tmp
    typecode = gpuarray_register_type(t, NULL)
    if typecode == -1:
        free(tmp)
        free(t)
        raise RuntimeError, "Could not register type"
    NP_TO_TYPE[dtype] = typecode
    TYPE_TO_NP[typecode] = dtype

cdef np.dtype typecode_to_dtype(int typecode):
    res = TYPE_TO_NP.get(typecode, None)
    if res is not None:
        return res
    else:
        raise NotImplementedError, "TODO"

# This is a stupid wrapper to avoid the extra argument introduced by having
# dtype_to_typecode declared 'cpdef'.
cdef int get_typecode(dtype) except -1:
    return dtype_to_typecode(dtype)

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
    raise ValueError, "don't know how to convert to dtype: %s"%(dtype,)

def dtype_to_ctype(dtype):
    """
    dtype_to_ctype(dtype)

    Return the C name for a type.

    :param dtype: type to get the name for
    :type dtype: numpy.dtype
    :rtype: string
    """
    cdef int typecode = dtype_to_typecode(dtype)
    cdef const gpuarray_type *t = gpuarray_get_type(typecode)
    if t.cluda_name == NULL:
        raise ValueError, "No mapping for %s"%(dtype,)
    return t.cluda_name

cdef ga_order to_ga_order(ord) except <ga_order>-2:
    if ord == "C" or ord == "c":
        return GA_C_ORDER
    elif ord == "A" or ord == "a" or ord is None:
        return GA_ANY_ORDER
    elif ord == "F" or ord == "f":
        return GA_F_ORDER
    else:
        raise ValueError, "Valid orders are: 'A' (any), 'C' (C), 'F' (Fortran)"

class GpuArrayException(Exception):
    """
    Exception used for most errors related to libgpuarray.
    """

class UnsupportedException(GpuArrayException):
    pass

cdef type get_exc(int errcode):
    if errcode == GA_VALUE_ERROR:
        return ValueError
    if errcode == GA_DEVSUP_ERROR:
        return UnsupportedException
    else:
        return GpuArrayException

cdef bint py_CHKFLAGS(GpuArray a, int flags):
    return GpuArray_CHKFLAGS(&a.ga, flags)

cdef bint py_ISONESEGMENT(GpuArray a):
    return GpuArray_ISONESEGMENT(&a.ga)

cdef int array_empty(GpuArray a, const gpuarray_buffer_ops *ops, void *ctx,
                     int typecode, unsigned int nd, const size_t *dims,
                     ga_order ord) except -1:
    cdef int err
    err = GpuArray_empty(&a.ga, ops, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise get_exc(err), Gpu_error(ops, ctx, err)

cdef int array_fromdata(GpuArray a, const gpuarray_buffer_ops *ops,
                        gpudata *data, size_t offset, int typecode,
                        unsigned int nd, const size_t *dims,
                        const ssize_t *strides, int writeable) except -1:
    cdef int err
    cdef void *ctx
    err = GpuArray_fromdata(&a.ga, ops, data, offset, typecode, nd, dims,
                            strides, writeable)
    if err != GA_NO_ERROR:
        ops.property(NULL, data, NULL, GA_BUFFER_PROP_CTX, &ctx)
        raise get_exc(err), Gpu_error(ops, ctx, err)

cdef int array_copy_from_host(GpuArray a, const gpuarray_buffer_ops *ops,
                              void *ctx, void *buf, int typecode,
                              unsigned int nd, const size_t *dims,
                              const ssize_t *strides) except -1:
    cdef int err
    err = GpuArray_copy_from_host(&a.ga, ops, ctx, buf, typecode, nd, dims,
                                  strides);
    if err != GA_NO_ERROR:
        raise get_exc(err), Gpu_error(ops, ctx, err)

cdef int array_view(GpuArray v, GpuArray a) except -1:
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_sync(GpuArray a) except -1:
    cdef int err
    err = GpuArray_sync(&a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_index(GpuArray r, GpuArray a, const ssize_t *starts,
                     const ssize_t *stops, const ssize_t *steps) except -1:
    cdef int err
    err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_setarray(GpuArray v, GpuArray a) except -1:
    cdef int err
    err = GpuArray_setarray(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&v.ga, err)

cdef int array_reshape(GpuArray res, GpuArray a, unsigned int nd,
                       const size_t *newdims, ga_order ord,
                       bint nocopy) except -1:
    cdef int err
    err = GpuArray_reshape(&res.ga, &a.ga, nd, newdims, ord, nocopy)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_transpose(GpuArray res, GpuArray a,
                         const unsigned int *new_axes) except -1:
    cdef int err
    err = GpuArray_transpose(&res.ga, &a.ga, new_axes)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_clear(GpuArray a) except -1:
    GpuArray_clear(&a.ga)

cdef bint array_share(GpuArray a, GpuArray b):
    return GpuArray_share(&a.ga, &b.ga)

cdef void *array_context(GpuArray a) except NULL:
    cdef void *res
    res = GpuArray_context(&a.ga)
    if res is NULL:
        raise GpuArrayException, "Invalid array or destroyed context"
    return res

cdef int array_move(GpuArray a, GpuArray src) except -1:
    cdef int err
    err = GpuArray_move(&a.ga, &src.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_write(GpuArray a, void *src, size_t sz) except -1:
    cdef int err
    err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_read(void *dst, size_t sz, GpuArray src) except -1:
    cdef int err
    err = GpuArray_read(dst, sz, &src.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&src.ga, err)

cdef int array_memset(GpuArray a, int data) except -1:
    cdef int err
    err = GpuArray_memset(&a.ga, data)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_copy(GpuArray res, GpuArray a, ga_order order) except -1:
    cdef int err
    err = GpuArray_copy(&res.ga, &a.ga, order)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_transfer(GpuArray res, GpuArray a, void *new_ctx,
                        const gpuarray_buffer_ops *new_ops,
                        bint may_share) except -1:
    cdef int err
    err = GpuArray_transfer(&res.ga, &a.ga, new_ctx, new_ops, may_share)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_split(_GpuArray **res, GpuArray a, size_t n, size_t *p,
                     unsigned int axis) except -1:
    cdef int err
    err = GpuArray_split(res, &a.ga, n, p, axis)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_concatenate(GpuArray r, const _GpuArray **a, size_t n,
                           unsigned int axis, int restype) except -1:
    cdef int err
    err = GpuArray_concatenate(&r.ga, a, n, axis, restype)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(a[0], err)

cdef const char *kernel_error(GpuKernel k, int err) except NULL:
    return Gpu_error(k.k.ops, kernel_context(k), err)

cdef int kernel_init(GpuKernel k, const gpuarray_buffer_ops *ops, void *ctx,
                     unsigned int count, const char **strs, const size_t *len,
                     const char *name, unsigned int argcount, const int *types,
                     int flags, char **err_str) except -1:
    cdef int err
    err = GpuKernel_init(&k.k, ops, ctx, count, strs, len, name, argcount,
                          types, flags, err_str)
    if err != GA_NO_ERROR:
        raise get_exc(err), Gpu_error(ops, ctx, err)

cdef int kernel_clear(GpuKernel k) except -1:
    GpuKernel_clear(&k.k)

cdef void *kernel_context(GpuKernel k) except NULL:
    cdef void *res
    res = GpuKernel_context(&k.k)
    if res is NULL:
        raise GpuArrayException, "Invalid kernel or destroyed context"
    return res

cdef int kernel_call(GpuKernel k, size_t n, size_t ls, size_t gs,
                     void **args) except -1:
    cdef int err
    err = GpuKernel_call(&k.k, n, ls, gs, args)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_call2(GpuKernel k, size_t n[2], size_t ls[2], size_t gs[2],
                     void **args) except -1:
    cdef int err
    err = GpuKernel_call2(&k.k, n, ls, gs, args)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_binary(GpuKernel k, size_t *sz, void **bin) except -1:
    cdef int err
    err = GpuKernel_binary(&k.k, sz, bin)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_property(GpuKernel k, int prop_id, void *res) except -1:
    cdef int err
    err = k.k.ops.property(NULL, NULL, k.k.k, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef GpuContext pygpu_default_context():
    return default_context

cdef GpuContext default_context = None

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1:
    cdef int err
    err = c.ops.property(c.ctx, NULL, NULL, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), Gpu_error(c.ops, c.ctx, err)

cdef const gpuarray_buffer_ops *get_ops(kind) except NULL:
    cdef const gpuarray_buffer_ops *res
    res = gpuarray_get_ops(kind)
    if res == NULL:
        raise RuntimeError, "Unsupported kind: %s" % (kind,)
    return res

cdef ops_kind(const gpuarray_buffer_ops *ops):
    if ops == gpuarray_get_ops("opencl"):
        return "opencl"
    if ops == gpuarray_get_ops("cuda"):
        return "cuda"
    raise RuntimeError, "Unknown ops vector"

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
    global default_context
    default_context = ctx

def get_default_context():
    """
    get_default_context()

    Return the currently defined default context (or `None`).
    """
    return default_context

cdef GpuContext ensure_context(GpuContext c):
    global default_context
    if c is None:
        if default_context is None:
            raise TypeError, "No context specified."
        return default_context
    return c

cdef bint pygpu_GpuArray_Check(object o):
    return isinstance(o, GpuArray)

cdef GpuContext pygpu_init(dev):
    if dev.startswith('cuda'):
        kind = "cuda"
        if dev[4:] == '':
            devnum = -1
        else:
            devnum = int(dev[4:])
    elif dev.startswith('opencl'):
        kind = "opencl"
        devspec = dev[6:].split(':')
        if len(devspec) < 2:
            raise ValueError, "OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev
        if not devspec[0].isdigit() or not devspec[1].isdigit():
            raise ValueError, "OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev
        else:
            devnum = int(devspec[0]) << 16 | int(devspec[1])
    else:
        raise ValueError, "Unknown device format:" + dev
    return GpuContext(kind, devnum)

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
    If you don't specify a number (e.g. 'cuda') the ambient context,
    which must have been initialized prior to this call, will be used.

    For opencl the device id is the platform number, a colon (:) and
    the device number.  There are no widespread and/or easy way to
    list available platforms and devices.  You can experiement with
    the values, unavaiable ones will just raise an error, and there
    are no gaps in the valid numbers.
    """
    return pygpu_init(dev)

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

cdef GpuArray pygpu_zeros(unsigned int nd, const size_t *dims, int typecode,
                          ga_order order, GpuContext context, type cls):
    cdef GpuArray res
    res = pygpu_empty(nd, dims, typecode, order, context, cls)
    array_memset(res, 0)
    return res

cdef GpuArray pygpu_empty(unsigned int nd, const size_t *dims, int typecode,
                          ga_order order, GpuContext context, type cls):
    cdef GpuArray res

    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
    array_empty(res, context.ops, context.ctx, typecode, nd, dims, order)
    return res

cdef GpuArray pygpu_fromhostdata(void *buf, int typecode, unsigned int nd,
                                 const size_t *dims, const ssize_t *strides,
                                 GpuContext context, type cls):
    cdef GpuArray res
    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
    array_copy_from_host(res, context.ops, context.ctx, buf, typecode, nd,
                         dims, strides)
    return res

cdef GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
                                unsigned int nd, const size_t *dims,
                                const ssize_t *strides, GpuContext context,
                                bint writable, object base, type cls):
    cdef GpuArray res

    res = new_GpuArray(cls, context, base)
    array_fromdata(res, context.ops, buf, offset, typecode, nd, dims,
                   strides, writable)
    return res


cdef GpuArray pygpu_copy(GpuArray a, ga_order ord):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, None)
    array_copy(res, a, ord)
    return res

cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
    array_move(a, src)
    return 0

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

    cdims = <size_t *>calloc(len(shape), sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(<unsigned int>len(shape), cdims,
                           dtype_to_typecode(dtype), to_ga_order(order),
                           context, cls)
    finally:
        free(cdims)

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
        only way to create gpudata pointers is through libgpuarray
        functions that aren't exposed at the python level. It can be
        used with the value of the `gpudata` attribute of an existing
        GpuArray.
    """
    cdef size_t *cdims = NULL
    cdef ssize_t *cstrides = NULL
    cdef unsigned int nd
    cdef size_t size
    cdef int typecode

    context = ensure_context(context)

    nd = <unsigned int>len(shape)
    if strides is not None and len(strides) != nd:
        raise ValueError, "strides must be the same length as shape"

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
            size = gpuarray_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
                strides[i] = size
                size *= cdims[i]

        return pygpu_fromgpudata(<gpudata *>data, offset, typecode, nd, cdims,
                                 cstrides, context, writable, base, cls)
    finally:
        free(cdims)
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, int ndmin=0,
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
            raise ValueError, "cannot copy an array to a different context"

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
                 (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
                 (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
                shp = arg.shape
                idx = (1,)*(ndmin-len(shp))
                shp = idx + shp
                arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
                arg = arg.view(cls)
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
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
                    context=arg.context)
        res.base = arg.base
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
            tmp = res
        array_move(tmp, arg)
        return res

    context = ensure_context(context)

    a = numpy.array(proto, dtype=dtype, order=order, ndmin=ndmin, copy=False)

    return pygpu_fromhostdata(np.PyArray_DATA(a), dtype_to_typecode(a.dtype),
                              np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
                              <ssize_t *>np.PyArray_STRIDES(a), context, cls)

cdef class GpuContext:
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
    options for libgpuarray.

    If you want an alternative interface check :meth:`~pygpu.gpuarray.init`.
    """
    def __dealloc__(self):
        if self.ctx != NULL:
            self.ops.buffer_deinit(self.ctx)

    def __cinit__(self, kind, devno):
        cdef int err = GA_NO_ERROR
        cdef void *ctx
        self.ops = get_ops(kind)
        self.ctx = self.ops.buffer_init(devno, 0, &err)
        if (err != GA_NO_ERROR):
            if err == GA_VALUE_ERROR:
                raise get_exc(err), "No device %d"%(devno,)
            else:
                raise get_exc(err), self.ops.ctx_error(NULL)

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

    property maxgsize:
        "Maximum group size for kernel calls"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE, &res)
            return res

    property bin_id:
        "Binary compatibility id"
        def __get__(self):
            cdef const char *res
            ctx_property(self, GA_CTX_PROP_BIN_ID, &res)
            return res;

cdef class flags(object):
    cdef int fl

    def __cinit__(self, fl):
        self.fl = fl

    def __getitem__(self, idx):
        cdef const char *key
        cdef size_t n
        cdef char c

        if isinstance(idx, unicode):
            idx = idx.encode('UTF-8')
        if isinstance(idx, bytes):
            key = idx
            n = len(idx)
        else:
            raise KeyError, "Unknown flag"
        if n == 1:
            c = key[0]
            if c == 'C':
                return self.c_contiguous
            elif c == 'F':
                return self.f_contiguous
            elif c == 'W':
                return self.writeable
            elif c == 'B':
                return self.behaved
            elif c == 'O':
                return self.owndata
            elif c == 'A':
                return self.aligned
            elif c == 'U':
                return self.updateifcopy
        elif n == 2:
            if strncmp(key, "CA", n) == 0:
                return self.carray
            if strncmp(key, "FA", n) == 0:
                return self.farray
        elif n == 3:
            if strncmp(key, "FNC", n) == 0:
                return self.fnc
        elif n == 4:
            if strncmp(key, "FORC", n) == 0:
                return self.forc
        elif n == 6:
            if strncmp(key, "CARRAY", n) == 0:
                return self.carray
            if strncmp(key, "FARRAY", n) == 0:
                return self.farray
        elif n == 7:
            if strncmp(key, "FORTRAN", n) == 0:
                return self.fortran
            if strncmp(key, "BEHAVED", n) == 0:
                return self.behaved
            if strncmp(key, "OWNDATA", n) == 0:
                return self.owndata
            if strncmp(key, "ALIGNED", n) == 0:
                return self.aligned
        elif n == 9:
            if strncmp(key, "WRITEABLE", n) == 0:
                return self.writeable
        elif n == 10:
            if strncmp(key, "CONTIGUOUS", n) == 0:
                return self.c_contiguous
        elif n == 12:
            if strncmp(key, "UPDATEIFCOPY", n) == 0:
                return self.updateifcopy
            if strncmp(key, "C_CONTIGUOUS", n) == 0:
                return self.c_contiguous
            if strncmp(key, "F_CONTIGUOUS", n) == 0:
                return self.f_contiguous

        raise KeyError, "Unknown flag"

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
                         for name in ["c_contiguous", "f_contiguous",
                                      "owndata", "writeable", "aligned",
                                      "updateifcopy"])

    def __richcmp__(self, other, int op):
        cdef flags a
        cdef flags b
        if not isinstance(self, flags) or not isinstance(other, flags):
            return NotImplemented
        a = self
        b = other
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
            return a.fl != b.fl
        raise TypeError, "undefined comparison for flag object"

    property c_contiguous:
        def __get__(self):
            return bool(self.fl & GA_C_CONTIGUOUS)

    property contiguous:
        def __get__(self):
            return self.c_contiguous

    property f_contiguous:
        def __get__(self):
            return bool(self.fl & GA_F_CONTIGUOUS)

    property fortran:
        def __get__(self):
            return self.f_contiguous

    property updateifcopy:
        # Not supported.
        def __get__(self):
            return False

    property owndata:
        # There is no equivalent for GpuArrays and it is always "True".
        def __get__(self):
            return True

    property aligned:
        def __get__(self):
            return bool(self.fl & GA_ALIGNED)

    property writeable:
        def __get__(self):
            return bool(self.fl & GA_WRITEABLE)

    property behaved:
        def __get__(self):
            return (self.fl & GA_BEHAVED) == GA_BEHAVED

    property carray:
        def __get__(self):
            return (self.fl & GA_CARRAY) == GA_CARRAY

    # Yes these are really defined like that according to numpy sources.
    # I don't know why.
    property forc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS or
                    (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

    property fnc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS and
                    not (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

    property farray:
        def __get__(self):
            return ((self.fl & GA_FARRAY) != 0 and
                    not ((self.fl & GA_C_CONTIGUOUS) != 0))

    property num:
        def __get__(self):
            return self.fl

cdef GpuArray new_GpuArray(type cls, GpuContext ctx, object base):
    cdef GpuArray res
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
        res = GpuArray.__new__(cls)
    res.base = base
    res.context = ctx
    return res

cdef GpuArray pygpu_view(GpuArray a, type cls):
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
    return res

cdef int pygpu_sync(GpuArray a) except -1:
    array_sync(a)
    return 0

cdef GpuArray pygpu_empty_like(GpuArray a, ga_order ord, int typecode):
    cdef GpuArray res

    if ord == GA_ANY_ORDER:
        if py_CHKFLAGS(a, GA_F_CONTIGUOUS) and \
                not py_CHKFLAGS(a, GA_C_CONTIGUOUS):
            ord = GA_F_ORDER
        else:
            ord = GA_C_ORDER

    if typecode == -1:
        typecode = a.ga.typecode

    res = new_GpuArray(type(a), a.context, None)
    array_empty(res, a.ga.ops, a.context.ctx, typecode,
                a.ga.nd, a.ga.dimensions, ord)
    return res

cdef np.ndarray pygpu_as_ndarray(GpuArray a):
    cdef np.ndarray res

    if not py_ISONESEGMENT(a):
        a = pygpu_copy(a, GA_ANY_ORDER)

    res = PyArray_Empty(a.ga.nd, <np.npy_intp *>a.ga.dimensions,
                        a.dtype, (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
                                  not py_CHKFLAGS(a, GA_C_CONTIGUOUS)))

    array_read(np.PyArray_DATA(res), np.PyArray_NBYTES(res), a)

    return res

cdef GpuArray pygpu_index(GpuArray a, const ssize_t *starts,
                          const ssize_t *stops, const ssize_t *steps):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
        raise IndexError, "index out of bounds"
    return res

cdef GpuArray pygpu_reshape(GpuArray a, unsigned int nd, const size_t *newdims,
                            ga_order ord, bint nocopy, int compute_axis):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
    if compute_axis >= nd:
        raise ValueError("You wanted us to compute the shape of a dimensions that don't exist")

    cdef size_t *cdims
    cdef size_t tot = 1
    for i in range(nd):
        d = newdims[i]
        if d != compute_axis:
            tot *= d
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"

    for i in range(nd):
        d = newdims[i]
        if i == compute_axis:
            d = a.size // tot

            if d * tot != a.size:
                raise GpuArrayException, "..."
        cdims[i] = d

    array_reshape(res, a, nd, cdims, ord, nocopy)
    return res


cdef GpuArray pygpu_transpose(GpuArray a, const unsigned int *newaxes):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    array_transpose(res, a, newaxes)
    return res

cdef GpuArray pygpu_transfer(GpuArray a, GpuContext new_ctx, bint may_share):
    cdef GpuArray res
    res = new_GpuArray(type(a), new_ctx, None)
    array_transfer(res, a, new_ctx.ctx, new_ctx.ops, may_share)
    return res

def _split(GpuArray a, ind, unsigned int axis):
    cdef list r = [None] * (len(ind) + 1)
    cdef Py_ssize_t i
    if not axis < a.ga.nd:
        raise ValueError, "split on non-existant axis"
    cdef size_t m = a.ga.dimensions[axis]
    cdef size_t v
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t) * len(ind))
    if p == NULL:
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *) * len(r))
    if rs == NULL:
        PyMem_Free(p)
        raise MemoryError()
    try:
        for i in range(len(r)):
            r[i] = new_GpuArray(type(a), a.context, a)
            rs[i] = &(<GpuArray>r[i]).ga
        for i in range(len(ind)):
            v = ind[i]
            # cap the values to the end of the array
            p[i] = v if v < m else m
        array_split(rs, a, len(ind), p, axis)
        return r
    finally:
        PyMem_Free(p)
        PyMem_Free(rs)

cdef GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
                                unsigned int axis, int restype,
                                type cls, GpuContext context):
    cdef res = new_GpuArray(cls, context, None)
    array_concatenate(res, a, n, axis, restype)
    return res

def _concatenate(list al, unsigned int axis, int restype, type cls,
                 GpuContext context):
    cdef Py_ssize_t i
    context = ensure_context(context)
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *) * len(al))
    if als == NULL:
        raise MemoryError()
    try:
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
                raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
        return pygpu_concatenate(als, len(al), axis, restype, cls, context)
    finally:
        PyMem_Free(als)

cdef class GpuArray:
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
    def __dealloc__(self):
        array_clear(self)

    def __cinit__(self):
        memset(&self.ga, 0, sizeof(_GpuArray))

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
                        ssize_t *stop, ssize_t *step):
        cdef Py_ssize_t dummy
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
                k += self.ga.dimensions[i]
            if k < 0 or k >= self.ga.dimensions[i]:
                raise IndexError, "index %d out of bounds" % (i,)
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
            raise IndexError, "cannot index with: %s" % (key,)

    def __array__(self):
        """
        __array__()

        Return a :class:`numpy.ndarray` with the same content.

        Automatically used by :meth:`numpy.asarray`.
        """
        return pygpu_as_ndarray(self)

    def _empty_like_me(self, dtype=None, order='C'):
        """
        _empty_like_me(dtype=None, order='C')

        Returns an empty (uninitialized) GpuArray with the same
        properties except if overridden by parameters.
        """
        cdef int typecode
        cdef GpuArray res

        if dtype is None:
            typecode = -1
        else:
            typecode = dtype_to_typecode(dtype)

        return pygpu_empty_like(self, to_ga_order(order), typecode)

    def copy(self, order='C'):
        """
        copy(order='C')

        Return a copy if this array.

        :param order: memory layout of the copy
        :type order: string
        """
        return pygpu_copy(self, to_ga_order(order))

    def transfer(self, GpuContext new_ctx, share=False):
        return pygpu_transfer(self, new_ctx, share)

    def __copy__(self):
        return pygpu_copy(self, GA_C_ORDER)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            return pygpu_copy(self, GA_C_ORDER)

    def sync(self):
        """
        sync()

        Wait for all pending operations on this array.

        This is done automatically when reading or writing from it,
        but can be useful as a separate operation for timings.
        """
        pygpu_sync(self)

    def view(self, type cls=GpuArray):
        """
        view(cls=GpuArray)

        Return a view of this array.

        :param cls: class of the view (must inherit from GpuArray)

        The returned array shares device data with this one and both
        will reflect changes made to the other.
        """
        return pygpu_view(self, cls)

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
        cdef int compute_axis
        nd = <unsigned int>len(shape)
        newdims = <size_t *>calloc(nd, sizeof(size_t))
        if newdims == NULL:
            raise MemoryError, "calloc"
        compute_axis = -1
        try:
            for i in range(nd):
                if shape[i] == -1:
                    assert compute_axis == -1
                    compute_axis = i
                    newdims[i] = 1
                else:
                    newdims[i] = shape[i]
            return pygpu_reshape(self, nd, newdims, to_ga_order(order), 0, compute_axis)
        finally:
            free(newdims)

    def transpose(self, *params):
        cdef unsigned int *new_axes
        cdef unsigned int i
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
        else:
            if len(params) != self.ga.nd:
                raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
            try:
                for i in range(self.ga.nd):
                    new_axes[i] = params[i]
                return pygpu_transpose(self, new_axes)
            finally:
                free(new_axes)

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
        else:
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
        cdef ssize_t *starts
        cdef ssize_t *stops
        cdef ssize_t *steps
        cdef unsigned int i
        cdef unsigned int d
        cdef unsigned int el

        if key is Ellipsis:
            return self
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
                return self
            else:
                raise IndexError, "0-d arrays can't be indexed"

        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
                raise MemoryError

            d = 0

            if isinstance(key, (tuple, list)):
                if Ellipsis in key:
                    # The following code replaces the first Ellipsis
                    # found in the key by a bunch of them depending on
                    # the number of dimensions.  As example, this
                    # allows indexing on the last dimension with
                    # a[..., 1:] on any array (including 1-dim).  This
                    # is also required for numpy compat.
                    el = key.index(Ellipsis)
                    if isinstance(key, tuple):
                        key = key[:el] + \
                              (Ellipsis,)*(self.ga.nd - (len(key) - 1)) + \
                              key[el+1:]
                    else:
                        key = key[:el] + \
                              [Ellipsis,]*(self.ga.nd - (len(key) - 1)) + \
                              key[el+1:]
                if len(key) > self.ga.nd:
                    raise IndexError, "too many indices"
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

            return pygpu_index(self, starts, stops, steps)
        finally:
            free(starts)
            free(stops)
            free(steps)

    def __setitem__(self, idx, v):
        cdef GpuArray tmp = self.__getitem__(idx)
        cdef GpuArray gv = asarray(v, dtype=self.dtype,
                                   context=self.context)

        array_setarray(tmp, gv)

    def __hash__(self):
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
            raise ValueError, "Thruth value of array with more than one element is ambiguous"

    property shape:
        "shape of this ndarray (tuple)"
        def __get__(self):
            cdef unsigned int i
            res = [None] * self.ga.nd
            for i in range(self.ga.nd):
                res[i] = self.ga.dimensions[i]
            return tuple(res)

        def __set__(self, newshape):
            # We support -1 only in a call to reshape
            cdef size_t *newdims
            cdef unsigned int nd
            cdef unsigned int i
            cdef GpuArray res
            nd = <unsigned int>len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
                raise MemoryError, "calloc"
            try:
                for i in range(nd):
                    newdims[i] = newshape[i]
                res = new_GpuArray(GpuArray, self.context, None)
                array_reshape(res, self, nd, newdims, GA_C_ORDER, 1)
            finally:
                free(newdims)
            # This is safe becase the reshape above is a nocopy one
            free(self.ga.dimensions)
            free(self.ga.strides)
            self.ga.dimensions = res.ga.dimensions
            self.ga.strides = res.ga.strides
            self.ga.nd = res.ga.nd
            res.ga.dimensions = NULL
            res.ga.strides = NULL
            res.ga.nd = 0
            array_clear(res)

    property T:
        def __get__(self):
            return pygpu_transpose(self, NULL)

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
        "The gpuarray typecode for the data type of the array"
        def __get__(self):
            return self.ga.typecode

    property itemsize:
        "The size of the base element."
        def __get__(self):
            return gpuarray_get_elsize(self.ga.typecode)

    property flags:
        """Return a flags object describing the properties of this array.

        This is mostly numpy-compatible with some exceptions:
          * Flags are always constant (numpy allows modification of certain flags in certain cicumstances).
          * OWNDATA is always True, since the data is refcounted in libgpuarray.
          * UPDATEIFCOPY is not supported, therefore always False.
        """
        def __get__(self):
            return flags(self.ga.flags)

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

        GpuKernel(source, name, types, context=None, cluda=True, have_double=False, have_small=False, have_complex=False, have_half=False)

    Compile a kernel on the device

    :param source: complete kernel source code
    :type source: string
    :param name: function name of the kernel
    :type name: string
    :param types: list of argument types
    :type types: list or tuple
    :param context: device on which the kernel is compiled
    :type context: GpuContext
    :param cluda: use cluda layer?
    :param have_double: ensure working doubles?
    :param have_small: ensure types smaller than float will work?
    :param have_complex: ensure complex types will work?
    :param have_half: ensure half-floats will work?
    :param binary: kernel is pre-compiled binary blob?
    :param ptx: kernel is PTX code?
    :param cuda: kernel is cuda code?
    :param opencl: kernel is opencl code?

    The kernel function is retrieved using the provided `name` which
    must match what you named your kernel in `source`.  You can safely
    reuse the same name multiple times.

    .. note::

        With the cuda backend, unless you use `cluda=True`, you must
        either pass the mangled name of your kernel or declare the
        function 'extern "C"', because cuda uses a C++ compiler
        unconditionally.

    The `have_*` parameter are there to tell libgpuarray that we need
    the particular type or feature to work for this kernel.  If the
    request can't be satified a
    :class:`~pygpu.gpuarray.UnsupportedException` will be raised in the
    constructor.

    .. warning::

        If you do not set the `have_` flags properly, you will either
        get a device-specific error (the good case) or silent
        completly bogus data (the bad case).

    Once you have the kernel object you can simply call it like so::

        k = GpuKernel(...)
        k(param1, param2, n=n)

    where `n` is the minimum number of threads to run.  libgpuarray
    will try to stay close to this number but may run a few more
    threads to match the hardware preferred multiple and stay
    efficient.  You should watch out for this in your code and make
    sure to test against the size of your data.

    If you want more control over thread allocation you can use the
    `ls` and `gs` parameters like so::

        k = GpuKernel(...)
        k(param1, param2, ls=ls, gs=gs)

    If you choose to use this interface, make sure to stay within the
    limits of `k.maxlsize` and `ctx.maxgsize` or the call will fail.
    """
    def __dealloc__(self):
        free(self.callbuf)
        kernel_clear(self)

    def __cinit__(self, source, name, types, GpuContext context=None,
                  cluda=True, have_double=False, have_small=False,
                  have_complex=False, have_half=False, binary=False,
                  ptx=False, cuda=False, opencl=False, *a, **kwa):
        cdef const char *s[1]
        cdef size_t l
        cdef unsigned int numargs
        cdef unsigned int i
        cdef int *_types
        cdef const gpuarray_buffer_ops *ops
        cdef int flags = 0
        cdef char *err_str=NULL

        if not isinstance(source, (str, unicode)):
            raise TypeError, "Expected a string for the kernel source"
        if not isinstance(name, (str, unicode)):
            raise TypeError, "Expected a string for the kernel name"

        self.context = ensure_context(context)

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
        if binary:
            flags |= GA_USE_BINARY
        if ptx:
            flags |= GA_USE_PTX
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL

        s[0] = source
        if False:  ## TODO : remove if kernel printout remains in gpuarray_buffer_opencl.c
          print "<gpuarray-source>\n"
          for line_num, line_contents in enumerate(source.split("\n")): print "%04d %s" % (int(line_num),line_contents)
          print "</gpuarray-source>"
        l = len(source)
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
            raise MemoryError
        try:
            for i in range(numargs):
                if (types[i] == GpuArray):
                    _types[i] = GA_BUFFER
                else:
                    _types[i] = dtype_to_typecode(types[i])
                self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
                if self.callbuf[i] == NULL:
                    raise MemoryError
            kernel_init(self, self.context.ops, self.context.ctx, 1, s, &l,
                        name, numargs, _types, flags, &err_str)
        finally:
            if err_str != NULL:
                print "gpuarray.pyx PRINTING err_str\n"
                print err_str ## TODO : Pass up further...
                free(err_str)
            free(_types)

    def __call__(self, *args, n=None, ls=None, gs=None):
        if n == None and (ls == None or gs == None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, ls, gs, args)

    cdef do_call(self, py_n, py_ls, py_gs, py_args):
        cdef size_t _n[2]
        cdef size_t _gs[2]
        cdef size_t _ls[2]
        cdef size_t *n
        cdef size_t *gs
        cdef size_t *ls
        cdef size_t tmp
        cdef const int *types
        cdef unsigned int numargs
        cdef unsigned int i

        if py_n is None:
            n = NULL
        else:
            if isinstance(py_n, int):
                _n[0] = py_n
                _n[1] = 1
            elif isinstance(py_n, (list, tuple)):
                if len(py_n) != 2:
                    raise ValueError, "n is not a len() 2 list"
                _n[0] = py_n[0]
                _n[1] = py_n[1]
            else:
                raise TypeError, "n is not int or list"
            n = _n

        if py_ls is None:
            ls = NULL
        else:
            if isinstance(py_ls, int):
                _ls[0] = py_ls
                _ls[1] = 1
            elif isinstance(py_ls, (list, tuple)):
                if len(py_ls) != 2:
                    raise ValueError, "ls is not a len() 2 list"
                _ls[0] = py_ls[0]
                _ls[1] = py_ls[1]
            else:
                raise TypeError, "ls is not int or list"
            ls = _ls

        if py_gs is None:
            gs = NULL
        else:
            if isinstance(py_gs, int):
                _gs[0] = py_gs
                _gs[1] = 1
            elif isinstance(py_gs, (list, tuple)):
                if len(py_gs) != 2:
                    raise ValueError, "gs is not a len() 2 list"
                _gs[0] = py_gs[0]
                _gs[1] = py_gs[1]
            else:
                raise TypeError, "gs is not int or list"
            gs = _gs

        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        kernel_call2(self, n, ls, gs, self.callbuf)

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
                raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>&(<GpuArray>o).ga
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
            (<unsigned long *>self.callbuf[index])[0] = o
        else:
            raise ValueError, "Bad typecode in _setarg (please report this, it is a bug)"

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

    property numargs:
        "Number of arguments to kernel"
        def __get__(self):
            cdef unsigned int res
            kernel_property(self, GA_KERNEL_PROP_NUMARGS, &res)
            return res

    property _binary:
        "Kernel compiled binary for the associated context."
        def __get__(self):
            cdef size_t sz
            cdef char *bin
            kernel_binary(self, &sz, <void **>&bin)
            try:
                return <bytes>bin[:sz]
            finally:
                free(bin)
