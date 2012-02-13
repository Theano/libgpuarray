cimport libc.stdio
from libc.stdlib cimport calloc, free
cimport numpy as np

from cpython cimport Py_INCREF

np.import_array()

cdef extern from "numpy/arrayobject.h":
    object _PyArray_Empty "PyArray_Empty" (int, np.npy_intp *, np.dtype, int)

# Numpy API steals dtype references and this breaks cython
cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d):
    Py_INCREF(c)
    return _PyArray_Empty(a, b, c, d)

cdef extern from *:
    void cifcuda "#ifdef WITH_CUDA //" ()
    void cifopencl "#ifdef WITH_OPENCL //" ()
    void cendif "#endif //" ()

cdef extern from "compyte_buffer.h":
    ctypedef struct gpudata:
        pass
    ctypedef struct gpukernel:
        pass

    ctypedef struct compyte_buffer_ops:
        void *buffer_init(int devno, int *ret)
        gpudata *buffer_alloc(void *ctx, size_t sz)
        void buffer_free(gpudata *d)
        int buffer_move(gpudata *dst, gpudata *src, size_t sz)
        int buffer_read(void *dst, gpudata *src, size_t sz)
        int buffer_write(gpudata *dst, void *src, size_t sz)
        int buffer_memset(gpudata *dst, int data, size_t sz)
        int buffer_offset(gpudata *buf, int offset)
        gpukernel *buffer_newkernel(void *ctx, unsigned int count,
                                    char **strings,
                                    size_t *lengths,
                                    char *fname)
        void buffer_freekernel(gpukernel *k)
        int buffer_setkernelarg(gpukernel *k, unsigned int index, size_t sz,
                                void *val)
        int buffer_setkernelargbuf(gpukernel *k, unsigned int index, gpudata *d)
        int buffer_callkernel(gpukernel *k, unsigned int gx, unsigned int gy,
                              unsigned int gz, unsigned int lx,
                              unsigned int ly, unsigned int lz)
        int buffer_elemwise(gpudata *input, gpudata *output, int intype,
                            int outtype, char *op, unsigned int nd,
                            size_t *dims, ssize_t *in_str, ssize_t *out_str)
        char *buffer_error()

    compyte_buffer_ops cuda_ops
    compyte_buffer_ops opencl_ops

    ctypedef struct _GpuArray "GpuArray":
        gpudata *data
        compyte_buffer_ops *ops
        size_t *dimensions
        ssize_t *strides
        size_t total_size
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

    cdef enum ga_error:
        GA_NO_ERROR, GA_MEMORY_ERROR, GA_VALUE_ERROR, GA_IMPL_ERROR,
        GA_INVALID_ERROR, GA_UNSUPPORTED_ERROR, GA_SYS_ERROR

    enum COMPYTE_TYPES:
        GA_FLOAT,
        GA_DOUBLE,
        GA_NBASE

    int GpuArray_empty(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)
    int GpuArray_zeros(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)
    int GpuArray_view(_GpuArray *v, _GpuArray *a)

    void GpuArray_clear(_GpuArray *a)
    
    int GpuArray_move(_GpuArray *dst, _GpuArray *src)
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz)
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src)
    int GpuArray_memset(_GpuArray *a, int data, size_t sz)

    char *GpuArray_error(_GpuArray *a, int err)
    
    void GpuArray_fprintf(libc.stdio.FILE *fd, _GpuArray *a)
    int GpuArray_is_c_contiguous(_GpuArray *a)
    int GpuArray_is_f_contiguous(_GpuArray *a)

import numpy

cdef int dtype_to_typecode(dtype):
    cdef int dnum
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, str):
        dtype = numpy.dtype(dtype)
    if isinstance(dtype, numpy.dtype):
        dnum = (<np.dtype>dtype).type_num
        if dnum < GA_NBASE:
            return dnum
    raise ValueError("don't know how to convert to dtype: %s"%(dtype,))

cdef int to_ga_order(ord):
    if ord == "C":
        return GA_C_ORDER
    elif ord == "A":
        return GA_ANY_ORDER
    elif ord == "F":
        return GA_F_ORDER
    else:
        raise ValueError("Valid orders are: 'A' (any), 'C' (C), 'F' (Fortran)")

class GpuArrayException(Exception):
    pass

cdef bint py_CHKFLAGS(GpuArray a, int flags):
    return GpuArray_CHKFLAGS(&a.ga, flags)

cdef bint py_ISONESEGMENT(GpuArray a):
    return GpuArray_ISONESEGMENT(&a.ga)

cdef _empty(GpuArray a, compyte_buffer_ops *ops, void *ctx, int typecode,
            unsigned int nd, size_t *dims, ga_order ord):
    cdef int err
    err = GpuArray_empty(&a.ga, ops, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef _zeros(GpuArray a, compyte_buffer_ops *ops, void *ctx, int typecode,
            unsigned int nd, size_t *dims, ga_order ord):
    cdef int err
    err = GpuArray_zeros(&a.ga, ops, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef _view(GpuArray v, GpuArray a):
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef _clear(GpuArray a):
    GpuArray_clear(&a.ga)

cdef _move(GpuArray a, GpuArray src):
    cdef int err
    err = GpuArray_move(&a.ga, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef _write(GpuArray a, void *src, size_t sz):
    cdef int err
    err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef _read(void *dst, size_t sz, GpuArray src):
    cdef int err
    err = GpuArray_read(dst, sz, &src.ga)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&src.ga, err))

cdef _memset(GpuArray a, int data, size_t sz):
    cdef int err
    err = GpuArray_memset(&a.ga, data, sz)
    if err != GA_NO_ERROR:
        raise GpuArrayException(GpuArray_error(&a.ga, err))

cdef compyte_buffer_ops *GpuArray_ops
cdef void *GpuArray_ctx

def set_kind_context(kind, size_t ctx):
    global GpuArray_ctx
    global GpuArray_ops
    GpuArray_ctx = <void *>ctx
    cifcuda()
    if kind == "cuda":
        GpuArray_ops = &cuda_ops
        return
    cendif()
    cifopencl()
    if kind == "opencl":
        GpuArray_ops = &opencl_ops
        return
    cendif()
    raise ValueError("Unknown kind")

cifopencl()
def init_opencl(int devno):
    cdef int err = GA_NO_ERROR
    cdef void *ctx
    ctx = opencl_ops.buffer_init(devno, &err)
    if (err != GA_NO_ERROR):
        raise GpuArrayException(opencl_ops.buffer_error())
    return <size_t>ctx
cendif()

cifcuda()
def init_cuda(int devno):
    cdef int err = GA_NO_ERROR
    cdef void *ctx
    ctx = cuda_ops.buffer_init(devno, &err)
    if (err != GA_NO_ERROR):
        raise GpuArrayException(cuda_ops.buffer_error())
    return <size_t>ctx
cendif()

def zeros(shape, dtype=GA_DOUBLE, order='A'):
    return GpuArray(shape, dtype=dtype, order=order, memset=0)

def empty(shape, dtype=GA_DOUBLE, order='A'):
    return GpuArray(shape, dtype=dtype, order=order)

cdef class GpuArray:
    cdef _GpuArray ga
    cdef object base

    def __dealloc__(self):
        _clear(self)

    def __cinit__(self, *a, **kwa):
        cdef size_t *cdims

        if len(a) == 1:
            proto = a[0]
            if isinstance(proto, np.ndarray):
                self.from_array(proto)
            elif isinstance(proto, GpuArray):
                if 'view' in kwa and kwa['view']:
                    self.make_view(proto)
                self.make_copy(proto, kwa.get('dtype',
                                              (<GpuArray>proto).ga.typecode),
                               to_ga_order(kwa.get('order', 'A')))
            elif isinstance(proto, (list, tuple)):
                cdims = <size_t *>calloc(len(proto), sizeof(size_t));
                if cdims == NULL:
                    raise MemoryError("could not allocate cdims")
                for i, d in enumerate(proto):
                    cdims[i] = d

                try:
                    self.make_empty(len(proto), cdims,
                                    kwa.get('dtype', GA_FLOAT),
                                    to_ga_order(kwa.get('order', 'A')))
                finally:
                    free(cdims)
                if 'memset' in kwa:
                    fill = kwa['memset']
                    if fill is not None:
                        self.memset(fill)
        else:
            raise ValueError("Cannot initialize from the given arguments")
    
    cdef make_empty(self, unsigned int nd, size_t *dims, dtype, ord):
        _empty(self, GpuArray_ops, GpuArray_ctx, dtype_to_typecode(dtype),
               nd, dims, ord)

    cdef make_copy(self, GpuArray other, dtype, ord):
        self.make_empty(other.ga.nd, other.ga.dimensions, dtype, ord)
        _move(self, other)

    cdef make_view(self, GpuArray other):
        _view(self, other)
        self.base = other

    def memset(self, value):
        _memset(self, value, self.ga.total_size)
    
    cdef from_array(self, np.ndarray a):
        if not np.PyArray_ISONESEGMENT(a):
            a = np.PyArray_ContiguousFromAny(a, np.PyArray_TYPE(a),
                                             np.PyArray_NDIM(a),
                                             np.PyArray_NDIM(a))

        if np.PyArray_ISFORTRAN(a) and not np.PyArray_ISCONTIGUOUS(a):
            order = GA_F_ORDER
        else:
            order = GA_C_ORDER
        self.make_empty(np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
                        a.dtype, order)

        _write(self, np.PyArray_DATA(a), self.ga.total_size)

    def __array__(self):
        cdef np.ndarray res
        
        if not py_ISONESEGMENT(self):
            self = self.copy()

        res = PyArray_Empty(self.ga.nd, <np.npy_intp *>self.ga.dimensions,
                            self.dtype, py_CHKFLAGS(self, GA_F_CONTIGUOUS) \
                                and not py_CHKFLAGS(self, GA_C_CONTIGUOUS))
        
        _read(np.PyArray_DATA(res),
              np.PyArray_SIZE(res)*np.PyArray_ITEMSIZE(res),
              self)
        
        return res

    def copy(self, dtype=None, order='A'):
        if dtype is None:
            dtype = self.ga.typecode
        return GpuArray(self, dtype, order)

    def view(self):
        return GpuArray(self, view=True)
        
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

    property strides:
        "data pointer strides (in bytes)"
        def __get__(self):
            cdef unsigned int i
            res = [None] * self.ga.nd
            for i in range(self.ga.nd):
                res[i] = self.ga.strides[i]
            return tuple(res)
        
    property ndim:
        def __get__(self):
            return self.ga.nd
        
    property dtype:
        def __get__(self):
            # XXX: will have to figure out the right numpy dtype
            if self.ga.typecode < GA_NBASE:
                return np.PyArray_DescrFromType(self.ga.typecode)
            else:
                raise NotImplementedError("TODO")
    
    property flags:
        def __get__(self):
            res = dict()
            res["C_CONTIGUOUS"] = py_CHKFLAGS(self, GA_C_CONTIGUOUS)
            res["F_CONTIGUOUS"] = py_CHKFLAGS(self, GA_F_CONTIGUOUS)
            res["WRITEABLE"] = py_CHKFLAGS(self, GA_WRITEABLE)
            res["ALIGNED"] = py_CHKFLAGS(self, GA_ALIGNED)
            res["UPDATEIFCOPY"] = False # Unsupported
            res["OWNDATA"] = py_CHKFLAGS(self, GA_OWNDATA)
            return res
