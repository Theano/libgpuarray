cimport libc

# This is used in a hack to silence some over-eager warnings.
cdef extern from *:
    ctypedef object slice_object "PySliceObject *"

cdef extern from "stdlib.h":
    void *memcpy(void *dst, void *src, size_t n)
    void *memset(void *b, int c, size_t sz)

cimport numpy as np

cdef extern from "numpy/arrayobject.h":
    object _PyArray_Empty "PyArray_Empty" (int, np.npy_intp *, np.dtype, int)

cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d)

cdef extern from "Python.h":
    int PySlice_GetIndicesEx(slice_object slice, Py_ssize_t length,
                             Py_ssize_t *start, Py_ssize_t *stop,
                             Py_ssize_t *step,
                             Py_ssize_t *slicelength) except -1

cdef extern from "compyte/types.h":
    ctypedef struct compyte_type:
        const char *cluda_name
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
    int GA_CTX_PROP_MAXGSIZE
    int GA_BUFFER_PROP_CTX
    int GA_KERNEL_PROP_CTX
    int GA_KERNEL_PROP_MAXLSIZE
    int GA_KERNEL_PROP_PREFLSIZE

    cdef enum ga_usefl:
        GA_USE_CLUDA, GA_USE_SMALL, GA_USE_DOUBLE, GA_USE_COMPLEX, GA_USE_HALF

    char *Gpu_error(compyte_buffer_ops *o, void *ctx, int err)
    compyte_buffer_ops *compyte_get_ops(const char *) nogil

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
                          size_t offset, int typecode, unsigned int nd, size_t \
*dims,
                          ssize_t *strides, int writable)
    int GpuArray_view(_GpuArray *v, _GpuArray *a)
    int GpuArray_sync(_GpuArray *a)
    int GpuArray_index(_GpuArray *r, _GpuArray *a, ssize_t *starts,
                       ssize_t *stops, ssize_t *steps)
    int GpuArray_setarray(_GpuArray *v, _GpuArray *a)
    int GpuArray_reshape(_GpuArray *res, _GpuArray *a, unsigned int nd,
                         size_t *newdims, ga_order ord, int nocopy)
    int GpuArray_transpose(_GpuArray *res, _GpuArray *a,
                           unsigned int *new_axes)

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
    void *compyte_get_extension(const char *) nogil

cdef api np.dtype typecode_to_dtype(int typecode)
cdef api int get_typecode(dtype) except -1
cpdef int dtype_to_typecode(dtype) except -1

cdef ga_order to_ga_order(ord) except <ga_order>-2

cdef bint py_CHKFLAGS(GpuArray a, int flags)
cdef bint py_ISONESEGMENT(GpuArray a)

cdef array_empty(GpuArray a, compyte_buffer_ops *ops, void *ctx, int typecode,
                 unsigned int nd, size_t *dims, ga_order ord)
cdef array_fromdata(GpuArray a, compyte_buffer_ops *ops, gpudata *data,
                    size_t offset, int typecode, unsigned int nd, size_t *dims,
                    ssize_t *strides, int writeable)
cdef array_view(GpuArray v, GpuArray a)
cdef array_sync(GpuArray a)
cdef array_index(GpuArray r, GpuArray a, ssize_t *starts, ssize_t *stops,
                 ssize_t *steps)
cdef array_setarray(GpuArray v, GpuArray a)
cdef array_reshape(GpuArray res, GpuArray a, unsigned int nd, size_t *newdims,
                   ga_order ord, int nocopy)
cdef array_transpose(GpuArray res, GpuArray a, unsigned int *new_axes)
cdef array_clear(GpuArray a)
cdef bint array_share(GpuArray a, GpuArray b)
cdef void *array_context(GpuArray a)
cdef array_move(GpuArray a, GpuArray src)
cdef array_write(GpuArray a, void *src, size_t sz)
cdef array_read(void *dst, size_t sz, GpuArray src)
cdef array_memset(GpuArray a, int data)
cdef array_copy(GpuArray res, GpuArray a, ga_order order)

cdef const char *kernel_error(GpuKernel k, int err)
cdef kernel_init(GpuKernel k, compyte_buffer_ops *ops, void *ctx,
                 unsigned int count, const char **strs, size_t *len,
                 char *name, int flags)
cdef kernel_clear(GpuKernel k)
cdef void *kernel_context(GpuKernel k)
cdef kernel_setarg(GpuKernel k, unsigned int index, int typecode, void *arg)
cdef kernel_setbufarg(GpuKernel k, unsigned int index, GpuArray a)
cdef kernel_call(GpuKernel k, size_t n, size_t ls, size_t gs)
cdef kernel_property(GpuKernel k, int prop_id, void *res)

cdef api GpuContext GpuArray_default_context()

cdef ctx_property(GpuContext c, int prop_id, void *res)
cdef compyte_buffer_ops *get_ops(kind) except NULL
cdef ops_kind(compyte_buffer_ops *ops)
cdef GpuContext ensure_context(GpuContext c)

cdef api class GpuContext [type GpuContextType, object GpuContextObject]:
    cdef compyte_buffer_ops *ops
    cdef void* ctx

cdef api GpuArray new_GpuArray(cls, GpuContext ctx)

cdef api class GpuArray [type GpuArrayType, object GpuArrayObject]:
    cdef _GpuArray ga
    cdef readonly GpuContext context
    cdef readonly object base
    cdef object __weakref__

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
                        ssize_t *stop, ssize_t *step)

    cpdef copy(self, order=*)

cdef api class GpuKernel [type GpuKernelType, object GpuKernelObject]:
    cdef _GpuKernel k
    cdef readonly GpuContext context

    cpdef setargs(self, args)
    cpdef setarg(self, unsigned int index, o)
    cdef _setarg(self, unsigned int index, np.dtype t, object o)
    cpdef call(self, size_t n, size_t ls, size_t gs)
