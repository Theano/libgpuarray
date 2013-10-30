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
        char *ctx_error(void *ctx)
        int property(void *c, gpudata *b, gpukernel *k, int prop_id, void *res)

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

    char *Gpu_error(const compyte_buffer_ops *o, void *ctx, int err)
    const compyte_buffer_ops *compyte_get_ops(const char *) nogil

cdef extern from "compyte/kernel.h":
    ctypedef struct _GpuKernel "GpuKernel":
        gpukernel *k
        const compyte_buffer_ops *ops

    int GpuKernel_init(_GpuKernel *k, const compyte_buffer_ops *ops, void *ctx,
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
        const compyte_buffer_ops *ops
        size_t offset
        size_t *dimensions
        ssize_t *strides
        unsigned int nd
        int flags
        int typecode

    cdef int GA_C_CONTIGUOUS
    cdef int GA_F_CONTIGUOUS
    cdef int GA_ALIGNED
    cdef int GA_WRITEABLE
    cdef int GA_BEHAVED
    cdef int GA_CARRAY
    cdef int GA_FARRAY

    bint GpuArray_CHKFLAGS(_GpuArray *a, int fl)
    bint GpuArray_ISONESEGMENT(_GpuArray *a)

    ctypedef enum ga_order:
        GA_ANY_ORDER, GA_C_ORDER, GA_F_ORDER

    int GpuArray_empty(_GpuArray *a, const compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)
    int GpuArray_fromdata(_GpuArray *a, const compyte_buffer_ops *ops,
                          gpudata *data, size_t offset, int typecode,
                          unsigned int nd, const size_t *dims,
                          const ssize_t *strides, int writable)
    int GpuArray_copy_from_host(_GpuArray *a, const compyte_buffer_ops *ops,
                            void *ctx, void *buf, int typecode,
                            unsigned int nd, const size_t *dims,
                            const ssize_t *strides)
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
    bint GpuArray_is_c_contiguous(_GpuArray *a)
    bint GpuArray_is_f_contiguous(_GpuArray *a)

cdef extern from "compyte/extension.h":
    void *compyte_get_extension(const char *) nogil
    cdef int COMPYTE_CUDA_CTX_NOFREE

cdef api np.dtype typecode_to_dtype(int typecode)
cdef api int get_typecode(dtype) except -1
cpdef int dtype_to_typecode(dtype) except -1

cdef ga_order to_ga_order(ord) except <ga_order>-2

cdef bint py_CHKFLAGS(GpuArray a, int flags)
cdef bint py_ISONESEGMENT(GpuArray a)

cdef int array_empty(GpuArray a, const compyte_buffer_ops *ops, void *ctx,
                     int typecode, unsigned int nd, size_t *dims,
                     ga_order ord) except -1
cdef int array_fromdata(GpuArray a, const compyte_buffer_ops *ops,
                        gpudata *data, size_t offset, int typecode,
                        unsigned int nd, const size_t *dims,
                        const ssize_t *strides, int writeable) except -1
cdef int array_copy_from_host(GpuArray a, const compyte_buffer_ops *ops,
                              void *ctx, void *buf, int typecode,
                              unsigned int nd, const size_t *dims,
                              const ssize_t *strides) except -1
cdef int array_view(GpuArray v, GpuArray a) except -1
cdef int array_sync(GpuArray a) except -1
cdef int array_index(GpuArray r, GpuArray a, ssize_t *starts, ssize_t *stops,
                     ssize_t *steps) except -1
cdef int array_setarray(GpuArray v, GpuArray a) except -1
cdef int array_reshape(GpuArray res, GpuArray a, unsigned int nd,
                       const size_t *newdims, ga_order ord,
                       bint nocopy) except -1
cdef int array_transpose(GpuArray res, GpuArray a,
                         unsigned int *new_axes) except -1
cdef int array_clear(GpuArray a) except -1
cdef bint array_share(GpuArray a, GpuArray b)
cdef void *array_context(GpuArray a) except NULL
cdef int array_move(GpuArray a, GpuArray src) except -1
cdef int array_write(GpuArray a, void *src, size_t sz) except -1
cdef int array_read(void *dst, size_t sz, GpuArray src) except -1
cdef int array_memset(GpuArray a, int data) except -1
cdef int array_copy(GpuArray res, GpuArray a, ga_order order) except -1

cdef const char *kernel_error(GpuKernel k, int err) except NULL
cdef int kernel_init(GpuKernel k, const compyte_buffer_ops *ops, void *ctx,
                     unsigned int count, const char **strs, size_t *len,
                     char *name, int flags) except -1
cdef int kernel_clear(GpuKernel k) except -1
cdef void *kernel_context(GpuKernel k) except NULL
cdef int kernel_setarg(GpuKernel k, unsigned int index, int typecode,
                       void *arg) except -1
cdef int kernel_setbufarg(GpuKernel k, unsigned int index,
                          GpuArray a) except -1
cdef int kernel_call(GpuKernel k, size_t n, size_t ls, size_t gs) except -1
cdef int kernel_property(GpuKernel k, int prop_id, void *res) except -1

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1
cdef const compyte_buffer_ops *get_ops(kind) except NULL
cdef ops_kind(const compyte_buffer_ops *ops)
cdef GpuContext ensure_context(GpuContext c)

cdef api GpuContext pygpu_default_context()

cdef api GpuContext pygpu_init(object dev)

cdef api GpuArray pygpu_zeros(unsigned int nd, size_t *dims, int typecode,
                              ga_order order, GpuContext context, type cls)
cdef api GpuArray pygpu_empty(unsigned int nd, size_t *dims, int typecode,
                              ga_order order, GpuContext context, type cls)
cdef api GpuArray pygpu_fromhostdata(void *buf, int typecode, unsigned int nd,
                                     const size_t *dims,
                                     const ssize_t *strides,
                                     GpuContext context, type cls)

cdef api GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
                                    unsigned int nd, const size_t *dims,
                                    const ssize_t *strides, GpuContext context,
                                    bint writable, object base, type cls)

cdef api GpuArray pygpu_copy(GpuArray a, ga_order ord)

cdef api GpuArray pygpu_view(GpuArray a, type cls)

cdef api int pygpu_sync(GpuArray a) except -1

cdef api GpuArray pygpu_empty_like(GpuArray a, ga_order ord, int typecode)

cdef api np.ndarray pygpu_as_ndarray(GpuArray a)

cdef api GpuArray pygpu_index(GpuArray a, const ssize_t *starts,
                              const ssize_t *stops, const ssize_t *steps)

cdef api GpuArray pygpu_reshape(GpuArray a, unsigned int nd,
                                const size_t *newdims, ga_order ord,
                                bint nocopy, int compute_axis)
cdef api GpuArray pygpu_transpose(GpuArray a, const unsigned int *newaxes)

cdef api class GpuContext [type PyGpuContextType, object PyGpuContextObject]:
    cdef const compyte_buffer_ops *ops
    cdef void* ctx

cdef GpuArray new_GpuArray(type cls, GpuContext ctx, object base)

cdef api class GpuArray [type PyGpuArrayType, object PyGpuArrayObject]:
    cdef _GpuArray ga
    cdef readonly GpuContext context
    cdef readonly object base
    cdef object __weakref__

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
                        ssize_t *stop, ssize_t *step)

cdef api class GpuKernel [type PyGpuKernelType, object PyGpuKernelObject]:
    cdef _GpuKernel k
    cdef readonly GpuContext context
    cdef object __weakref__

    cpdef setargs(self, args)
    cpdef setarg(self, unsigned int index, o)
    cdef _setarg(self, unsigned int index, np.dtype t, object o)
    cpdef call(self, size_t n, size_t ls, size_t gs)
