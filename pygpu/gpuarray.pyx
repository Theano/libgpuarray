cimport libc.stdio
from libc.stdlib cimport malloc, calloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport strncmp

cimport numpy as np
import numpy as np

from cpython cimport Py_INCREF, PyNumber_Index
from cpython.object cimport Py_EQ, Py_NE

def api_version():
    # major, minor, py
    return (gpuarray_api_major, gpuarray_api_minor, 0)

np.import_array()

# to export the numeric value
SIZE = GA_SIZE
SSIZE = GA_SSIZE

# Numpy API steals dtype references and this breaks cython
cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d):
    Py_INCREF(c)
    return _PyArray_Empty(a, b, c, d)

cdef bytes _s(s):
    if isinstance(s, unicode):
        return (<unicode>s).encode('ascii')
    if isinstance(s, bytes):
        return s
    raise TypeError("Expected a string")

def cl_wrap_ctx(size_t ptr):
    """
    cl_wrap_ctx(ptr)

    Wrap an existing OpenCL context (the cl_context struct) into a
    GpuContext class.
    """
    cdef gpucontext *(*cl_make_ctx)(void *, int)
    cdef GpuContext res
    cl_make_ctx = <gpucontext *(*)(void *, int)>gpuarray_get_extension("cl_make_ctx")
    if cl_make_ctx == NULL:
        raise RuntimeError, "cl_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
    res.ctx = cl_make_ctx(<void *>ptr, 0)
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
    cdef gpucontext *(*cuda_make_ctx)(void *, int)
    cdef int flags
    cdef GpuContext res
    cuda_make_ctx = <gpucontext *(*)(void *, int)>gpuarray_get_extension("cuda_make_ctx")
    if cuda_make_ctx == NULL:
        raise RuntimeError, "cuda_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
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
    np.dtype('float16'): GA_HALF,
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

# This function takes a flexible dtype as accepted by the functions of
# this module and ensures it becomes a numpy dtype.
cdef np.dtype dtype_to_npdtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, int):
        return typecode_to_dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        pass
    if isinstance(dtype, np.dtype):
        return dtype
    raise ValueError("data type not understood", dtype)

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
    try:
        dtype = np.dtype(dtype)
    except TypeError:
        pass
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
    cdef bytes res
    if t.cluda_name == NULL:
        raise ValueError, "No mapping for %s"%(dtype,)
    res = t.cluda_name
    return res.decode('ascii')

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

cdef int array_empty(GpuArray a, gpucontext *ctx,
                     int typecode, unsigned int nd, const size_t *dims,
                     ga_order ord) except -1:
    cdef int err
    err = GpuArray_empty(&a.ga, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int array_fromdata(GpuArray a,
                        gpudata *data, size_t offset, int typecode,
                        unsigned int nd, const size_t *dims,
                        const ssize_t *strides, int writeable) except -1:
    cdef int err
    err = GpuArray_fromdata(&a.ga, data, offset, typecode, nd, dims,
                            strides, writeable)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(gpudata_context(data), err)

cdef int array_copy_from_host(GpuArray a,
                              gpucontext *ctx, void *buf, int typecode,
                              unsigned int nd, const size_t *dims,
                              const ssize_t *strides) except -1:
    cdef int err
    with nogil:
        err = GpuArray_copy_from_host(&a.ga, ctx, buf, typecode, nd, dims,
                                      strides);
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int array_view(GpuArray v, GpuArray a) except -1:
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_sync(GpuArray a) except -1:
    cdef int err
    with nogil:
        err = GpuArray_sync(&a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_index(GpuArray r, GpuArray a, const ssize_t *starts,
                     const ssize_t *stops, const ssize_t *steps) except -1:
    cdef int err
    err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_take1(GpuArray r, GpuArray a, GpuArray i,
                     int check_err) except -1:
    cdef int err
    err = GpuArray_take1(&r.ga, &a.ga, &i.ga, check_err)
    if err != GA_NO_ERROR:
        if err == GA_VALUE_ERROR:
            raise IndexError, "Index out of bounds"
        raise get_exc(err), GpuArray_error(&r.ga, err)

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

cdef gpucontext *array_context(GpuArray a) except NULL:
    cdef gpucontext *res
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
    with nogil:
        err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_read(void *dst, size_t sz, GpuArray src) except -1:
    cdef int err
    with nogil:
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

cdef int array_transfer(GpuArray res, GpuArray a) except -1:
    cdef int err
    with nogil:
        err = GpuArray_transfer(&res.ga, &a.ga)
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
    return gpucontext_error(gpukernel_context(k.k.k), err)

cdef int kernel_init(GpuKernel k, gpucontext *ctx,
                     unsigned int count, const char **strs, const size_t *len,
                     const char *name, unsigned int argcount, const int *types,
                     int flags) except -1:
    cdef int err
    cdef char *err_str = NULL
    err = GpuKernel_init(&k.k, ctx, count, strs, len, name, argcount,
                          types, flags, &err_str)
    if err != GA_NO_ERROR:
        if err_str != NULL:
            try:
                py_err_str = err_str.decode('UTF-8')
            finally:
                free(err_str)
            raise get_exc(err), py_err_str
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int kernel_clear(GpuKernel k) except -1:
    GpuKernel_clear(&k.k)

cdef gpucontext *kernel_context(GpuKernel k) except NULL:
    cdef gpucontext *res
    res = GpuKernel_context(&k.k)
    if res is NULL:
        raise GpuArrayException, "Invalid kernel or destroyed context"
    return res

cdef int kernel_sched(GpuKernel k, size_t n, size_t *ls, size_t *gs) except -1:
    cdef int err
    err = GpuKernel_sched(&k.k, n, ls, gs)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_call(GpuKernel k, unsigned int n, const size_t *ls,
                     const size_t *gs, size_t shared, void **args) except -1:
    cdef int err
    err = GpuKernel_call(&k.k, n, ls, gs, shared, args)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_binary(GpuKernel k, size_t *sz, void **bin) except -1:
    cdef int err
    err = GpuKernel_binary(&k.k, sz, bin)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_property(GpuKernel k, int prop_id, void *res) except -1:
    cdef int err
    err = gpukernel_property(k.k.k, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef GpuContext pygpu_default_context():
    return default_context

cdef GpuContext default_context = None

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1:
    cdef int err
    err = gpucontext_property(c.ctx, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(c.ctx, err)

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

def count_platforms(kind):
    """Return number of host's platforms compatible with `kind`.
    """
    cdef unsigned int platcount
    cdef int err
    err = gpu_get_platform_count(_s(kind), &platcount)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(NULL, err)
    return platcount

def count_devices(kind, unsigned int platform):
    """Returns number of devices in host's `platform` compatible with `kind`.
    """
    cdef unsigned int devcount
    cdef int err
    err = gpu_get_device_count(_s(kind), platform, &devcount)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(NULL, err)
    return devcount

cdef GpuContext pygpu_init(dev, int flags):
    if dev.startswith('cuda'):
        kind = b"cuda"
        if dev[4:] == '':
            devnum = -1
        else:
            devnum = int(dev[4:])
    elif dev.startswith('opencl'):
        kind = b"opencl"
        devspec = dev[6:].split(':')
        if len(devspec) < 2:
            raise ValueError, "OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev
        if not devspec[0].isdigit() or not devspec[1].isdigit():
            raise ValueError, "OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev
        else:
            devnum = int(devspec[0]) << 16 | int(devspec[1])
    else:
        raise ValueError, "Unknown device format:" + dev
    return GpuContext(kind, devnum, flags)

def init(dev, sched='default', disable_alloc_cache=False, single_stream=False):
    """
    init(dev, sched='default', disable_alloc_cache=False, single_stream=False)

    Creates a context from a device specifier.

    :param dev: device specifier
    :type dev: string
    :param sched: optimize scheduling for which type of operation
    :type sched: {'default', 'single', 'multi'}
    :param disable_alloc_cache: disable allocation cache (if any)
    :type disable_alloc_cache: bool
    :param single_stream: enable single stream mode
    :type single_stream: bool
    :rtype: GpuContext

    Device specifiers are composed of the type string and the device
    id like so::

        "cuda0"
        "opencl0:1"

    For cuda the device id is the numeric identifier.  You can see
    what devices are available by running nvidia-smi on the machine.
    Be aware that the ordering in nvidia-smi might not correspond to
    the ordering in this library.  This is due to how cuda enumerates
    devices.  If you don't specify a number (e.g. 'cuda') the first
    available device will be selected according to the backend order.

    For opencl the device id is the platform number, a colon (:) and
    the device number.  There are no widespread and/or easy way to
    list available platforms and devices.  You can experiement with
    the values, unavaiable ones will just raise an error, and there
    are no gaps in the valid numbers.
    """
    cdef int flags = 0
    expected_version = -9997
    if gpuarray_api_major != expected_version or gpuarray_api_minor < 0:
        raise RuntimeError(
            "Pygpu was expecting libgpuarray version %d, but %d is available. "
            "Recompile it to avoid problems.",
            expected_version, gpuarray_api_major)
    if sched == 'single':
        flags |= GA_CTX_SINGLE_THREAD
    elif sched == 'multi':
        flags |= GA_CTX_MULTI_THREAD
    elif sched != 'default':
        raise TypeError('unexpected value for parameter sched: %s' % (sched,))
    if disable_alloc_cache:
        flags |= GA_CTX_DISABLE_ALLOCATION_CACHE
    if single_stream:
        flags |= GA_CTX_SINGLE_STREAM
    return pygpu_init(dev, flags)

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
                          ga_order order, GpuContext context, object cls):
    cdef GpuArray res
    res = pygpu_empty(nd, dims, typecode, order, context, cls)
    array_memset(res, 0)
    return res

cdef GpuArray pygpu_empty(unsigned int nd, const size_t *dims, int typecode,
                          ga_order order, GpuContext context, object cls):
    cdef GpuArray res

    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
    array_empty(res, context.ctx, typecode, nd, dims, order)
    return res

cdef GpuArray pygpu_fromhostdata(void *buf, int typecode, unsigned int nd,
                                 const size_t *dims, const ssize_t *strides,
                                 GpuContext context, object cls):
    cdef GpuArray res
    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
    array_copy_from_host(res, context.ctx, buf, typecode, nd,
                         dims, strides)
    return res

cdef GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
                                unsigned int nd, const size_t *dims,
                                const ssize_t *strides, GpuContext context,
                                bint writable, object base, object cls):
    cdef GpuArray res

    res = new_GpuArray(cls, context, base)
    array_fromdata(res, buf, offset, typecode, nd, dims,
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
    cdef unsigned int nd

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]

    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
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
    :param strides: strides for the results (C contiguous if not specified)
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

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]

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
                cstrides[i] = size
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

    a = numpy.array(proto, dtype=dtype_to_npdtype(dtype), order=order,
                    ndmin=ndmin, copy=False)

    return pygpu_fromhostdata(np.PyArray_DATA(a), dtype_to_typecode(a.dtype),
                              np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
                              <ssize_t *>np.PyArray_STRIDES(a), context, cls)

cdef void (*cuda_enter)(gpucontext *)
cdef void (*cuda_exit)(gpucontext *)

cuda_enter = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_enter")
cuda_exit = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_exit")

cdef class GpuContext:
    """
    Class that holds all the information pertaining to a context.

    .. code-block:: python

        GpuContext(kind, devno, flags)

    :param kind: module name for the context
    :type kind: string
    :param devno: device number
    :type devno: int
    :param flags: context flags
    :type flags: int

    The currently implemented modules (for the `kind` parameter) are
    "cuda" and "opencl".  Which are available depends on the build
    options for libgpuarray.

    The flag values are defined in the gpuarray/buffer.h header and
    are in the "Context flags" group.  If you want to use more than
    one value you must bitwise OR them together.

    If you want an alternative interface check :meth:`~pygpu.gpuarray.init`.
    """
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuContext object"

    def __cinit__(self, bytes kind, devno, int flags):
        cdef int err = GA_NO_ERROR
        cdef gpucontext *ctx
        self.kind = kind
        self.ctx = gpucontext_init(<char *>self.kind, devno, flags, &err)
        if (err != GA_NO_ERROR):
            if err == GA_VALUE_ERROR:
                raise get_exc(err), "No device %d"%(devno,)
            else:
                raise get_exc(err), gpucontext_error(NULL, err).decode('utf-8') + ": " + str(devno)

    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
        return self

    def __exit__(self, t, v, tb):
        cuda_exit(self.ctx)

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

    property total_gmem:
        "Total size of global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_TOTAL_GMEM, &res)
            return res

    property free_gmem:
        "Size of free global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_FREE_GMEM, &res)
            return res

    property maxlsize0:
        "Maximum local size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE0, &res)
            return res

    property maxlsize1:
        "Maximum local size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE1, &res)
            return res

    property maxlsize2:
        "Maximum local size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE2, &res)
            return res

    property maxgsize0:
        "Maximum global size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE0, &res)
            return res

    property maxgsize1:
        "Maximum global size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE1, &res)
            return res

    property maxgsize2:
        "Maximum global size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE2, &res)
            return res


cdef class flags(object):
    cdef int fl

    def __cinit__(self, fl):
        self.fl = fl

    def __reduce__(self):
        return (flags, (self.fl,))

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

cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
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

cdef GpuArray pygpu_view(GpuArray a, object cls):
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
    array_empty(res, a.context.ctx, typecode,
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
        if i != compute_axis:
            tot *= newdims[i]
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

cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
    array_transfer(res, a)
    return 0

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
            r[i] = new_GpuArray(type(a), a.context, a.base)
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
                                object cls, GpuContext context):
    cdef res = new_GpuArray(cls, context, None)
    array_concatenate(res, a, n, axis, restype)
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
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

cdef int (*cuda_get_ipc_handle)(gpudata *, GpuArrayIpcMemHandle *)
cdef gpudata *(*cuda_open_ipc_handle)(gpucontext *, GpuArrayIpcMemHandle *, size_t)

cuda_get_ipc_handle = <int (*)(gpudata *, GpuArrayIpcMemHandle *)>gpuarray_get_extension("cuda_get_ipc_handle")
cuda_open_ipc_handle = <gpudata *(*)(gpucontext *, GpuArrayIpcMemHandle *, size_t)>gpuarray_get_extension("cuda_open_ipc_handle")

def open_ipc_handle(GpuContext c, bytes hpy, size_t l):
    """
    Open an IPC handle to get a new GpuArray from it.

    :param c: context
    :param hpy: binary handle data received
    :param l: size of the referred memory block

    """
    cdef char *b
    cdef GpuArrayIpcMemHandle h
    cdef gpudata *d

    b = hpy
    memcpy(&h, b, sizeof(h))

    d = cuda_open_ipc_handle(c.ctx, &h, l)
    if d is NULL:
        raise GpuArrayException, "could not open handle"
    return <size_t>d

cdef class GpuArray:
    """
    Device array

    To create instances of this class use
    :meth:`~pygpu.gpuarray.zeros`, :meth:`~pygpu.gpuarray.empty` or
    :meth:`~pygpu.gpuarray.array`.  It cannot be instanciated
    directly.

    You can also subclass this class and make the module create your
    instances by passing the `cls` argument to any method that return a
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

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuArray object"

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

    def write(self, np.ndarray src not None):
        """Writes host's Numpy array to device's GpuArray.

        This method is as fast as or even faster than :ref:asarray, because it
        skips possible allocation of a buffer in device's memory. It uses this
        already allocated GpuArray buffer to contain `src` array from host's
        memory. It is required though that the GpuArray and the Numpy array are
        compatible in byte size and data type. It is also needed for the
        GpuArray to be well behaved and contiguous. If `src` is not aligned or
        compatible in contiguity it will be copied to a new Numpy array in order
        to be. It is allowed for this GpuArray and `src` to have different
        shapes.

        :param src: source array in host
        :type src: np.ndarray

        :raises ValueError: If this GpuArray is not compatible with `src` or
            if it is not well behaved or contiguous.

        """
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
            src = np.asarray(src, order='F')
        else:
            raise ValueError, "Destination GpuArray is not contiguous"
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_write(self, np.PyArray_DATA(src), sz)

    def read(self, np.ndarray dst not None):
        """Reads from this GpuArray into host's Numpy array.

        This method is as fast as or even faster than :ref:__array__ method and
        thus :ref:numpy.asarray. This is because it skips allocation of a new
        buffer in host's memory to contain device's GpuArray. It uses an
        existing Numpy ndarray as a buffer to get the GpuArray. It is required
        though that the GpuArray and the Numpy array to be compatible in byte
        size, contiguity and data type. It is also needed for `dst` to be
        writeable and properly aligned in host's memory and for `self` to be
        contiguous. It is allowed for this GpuArray and `dst` to have different
        shapes.

        :param dst: destination array in host
        :type dst: np.ndarray

        :raises ValueError: If this GpuArray is not compatible with `src` or
            if `dst` is not well behaved.

        """
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if not ((self.flags.c_contiguous and self.flags.aligned and dst.flags['C_CONTIGUOUS']) or \
                (self.flags.f_contiguous and self.flags.aligned and dst.flags['F_CONTIGUOUS'])):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_read(np.PyArray_DATA(dst), sz, self)

    def get_ipc_handle(self):
        cdef GpuArrayIpcMemHandle h
        cdef int err
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
        return res

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

    def transfer(self, GpuContext new_ctx):
        cdef GpuArray r
        if not GpuArray_ISONESEGMENT(&self.ga):
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
        r = pygpu_empty(self.ga.nd, self.ga.dimensions, self.ga.typecode,
                        GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
                        new_ctx, None)
        pygpu_transfer(r, self)  # Will raise an error if needed
        return r

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

    def view(self, object cls=GpuArray):
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

        try:
            nd = <unsigned int>len(shape)
        except TypeError:
            nd = 1
            shape = [shape]

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
            return pygpu_view(self, None)
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

    def take1(self, GpuArray idx):
        cdef GpuArray res
        cdef size_t odim
        if idx.ga.nd != 1:
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
        try:
            self.ga.dimensions[0] = idx.ga.dimensions[0]
            res = pygpu_empty_like(self, GA_C_ORDER, -1)
        finally:
            self.ga.dimensions[0] = odim
        array_take1(res, self, idx, 1)
        return res

    def __hash__(self):
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
            raise ValueError, "Truth value of array with more than one element is ambiguous"

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
        "Return a pointer to the raw backend object."
        def __get__(self):
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
            # structure.
            return <size_t>((<void **>self.ga.data)[0])

    def __str__(self):
        return str(numpy.asarray(self))

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
            return 'gpuarray.array(<content not available>)'



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
        cdef unsigned int numargs
        cdef int *types
        cdef unsigned int i
        cdef int res
        # We need to do all of this at the C level to avoid touching
        # python stuff that could be gone and to avoid exceptions
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
                return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
                return
            for i in range(numargs):
                if types[i] != GA_BUFFER:
                    free(self.callbuf[i])
            kernel_clear(self)
        free(self.callbuf)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuKernel object"

    def __cinit__(self, source, name, types, GpuContext context=None,
                  cluda=True, have_double=False, have_small=False,
                  have_complex=False, have_half=False, binary=False,
                  cuda=False, opencl=False, *a, **kwa):
        cdef const char *s[1]
        cdef size_t l
        cdef unsigned int numargs
        cdef unsigned int i
        cdef int *_types
        cdef int flags = 0

        source = _s(source)
        name = _s(name)

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
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL

        s[0] = source
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
            kernel_init(self, self.context.ctx, 1, s, &l,
                        name, numargs, _types, flags)
        finally:
            free(_types)

    def __call__(self, *args, n=None, ls=None, gs=None, shared=0):
        if n == None and (ls == None or gs == None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, ls, gs, args, shared)

    cdef do_call(self, py_n, py_ls, py_gs, py_args, size_t shared):
        cdef size_t n
        cdef size_t gs[3]
        cdef size_t ls[3]
        cdef size_t tmp
        cdef unsigned int nd
        cdef const int *types
        cdef unsigned int numargs
        cdef unsigned int i

        nd = 0

        if py_ls is None:
            ls[0] = 0
            nd = 1
        else:
            if isinstance(py_ls, int):
                ls[0] = py_ls
                nd = 1
            elif isinstance(py_ls, (list, tuple)):
                if len(py_ls) > 3:
                    raise ValueError, "ls is not of length 3 or less"
                nd = len(py_ls)

                if nd >= 3:
                    ls[2] = py_ls[2]
                if nd >= 2:
                    ls[1] = py_ls[1]
                if nd >= 1:
                    ls[0] = py_ls[0]
            else:
                raise TypeError, "ls is not int or list"

        if py_gs is None:
            if nd != 1:
                raise ValueError, "nd mismatch for gs (None)"
            gs[0] = 0
        else:
            if isinstance(py_gs, int):
                if nd != 1:
                    raise ValueError, "nd mismatch for gs (int)"
                gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
                if len(py_gs) < 3:
                    raise ValueError, "gs is not of length 3 or less"
                if len(py_ls) != nd:
                    raise ValueError, "nd mismatch for gs (tuple)"

                if nd >= 3:
                    gs[2] = py_gs[2]
                if nd >= 2:
                    gs[1] = py_gs[1]
                if nd >= 1:
                    gs[0] = py_gs[0]
            else:
                raise TypeError, "gs is not int or list"

        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
                raise ValueError, "n is specified and nd != 1"
            n = py_n
            kernel_sched(self, n, &ls[0], &gs[0])
        kernel_call(self, nd, ls, gs, shared, self.callbuf)

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
                raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
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
            raise ValueError("Bad typecode in _setarg: %d "
                             "(please report this, it is a bug)" % (typecode,))

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
