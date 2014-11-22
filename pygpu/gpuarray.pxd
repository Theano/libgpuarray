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

cdef extern from "gpuarray/types.h":
    ctypedef struct gpuarray_type:
        const char *cluda_name
        size_t size
        size_t align
        int typecode

    enum GPUARRAY_TYPES:
        GA_BUFFER,
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
        GA_SIZE,
        GA_NBASE

cdef extern from "gpuarray/util.h":
    int gpuarray_register_type(gpuarray_type *t, int *ret)
    size_t gpuarray_get_elsize(int typecode)
    gpuarray_type *gpuarray_get_type(int typecode)

cdef extern from "gpuarray/error.h":
    cdef enum ga_error:
        GA_NO_ERROR, GA_MEMORY_ERROR, GA_VALUE_ERROR, GA_IMPL_ERROR,
        GA_INVALID_ERROR, GA_UNSUPPORTED_ERROR, GA_SYS_ERROR, GA_RUN_ERROR,
        GA_DEVSUP_ERROR, GA_READONLY_ERROR, GA_WRITEONLY_ERROR, GA_BLAS_ERROR,
        GA_UNALIGNED_ERROR, GA_COPY_ERROR

cdef extern from "gpuarray/buffer.h":
    ctypedef struct gpudata:
        pass
    ctypedef struct gpukernel:
        pass

    ctypedef struct gpuarray_buffer_ops:
        void *buffer_init(int devno, int flags, int *ret)
        void buffer_deinit(void *ctx)
        char *ctx_error(void *ctx)
        int property(void *c, gpudata *b, gpukernel *k, int prop_id, void *res)

    int GA_CTX_PROP_DEVNAME
    int GA_CTX_PROP_MAXLSIZE
    int GA_CTX_PROP_LMEMSIZE
    int GA_CTX_PROP_NUMPROCS
    int GA_CTX_PROP_MAXGSIZE
    int GA_CTX_PROP_BIN_ID
    int GA_BUFFER_PROP_CTX
    int GA_KERNEL_PROP_CTX
    int GA_KERNEL_PROP_MAXLSIZE
    int GA_KERNEL_PROP_PREFLSIZE
    int GA_KERNEL_PROP_NUMARGS
    int GA_KERNEL_PROP_TYPES

    cdef enum ga_usefl:
        GA_USE_CLUDA, GA_USE_SMALL, GA_USE_DOUBLE, GA_USE_COMPLEX, GA_USE_HALF,
        GA_USE_BINARY, GA_USE_PTX, GA_USE_CUDA, GA_USE_OPENCL

    char *Gpu_error(const gpuarray_buffer_ops *o, void *ctx, int err)
    const gpuarray_buffer_ops *gpuarray_get_ops(const char *) nogil

cdef extern from "gpuarray/kernel.h":
    ctypedef struct _GpuKernel "GpuKernel":
        gpukernel *k
        const gpuarray_buffer_ops *ops

    int GpuKernel_init(_GpuKernel *k, const gpuarray_buffer_ops *ops, void *ctx,
                       unsigned int count, const char **strs,
                       const size_t *lens, const char *name,
                       unsigned int argcount, const int *types, int flags, char **err_str)
    void GpuKernel_clear(_GpuKernel *k)
    void *GpuKernel_context(_GpuKernel *k)
    int GpuKernel_call(_GpuKernel *, size_t n, size_t ls, size_t gs,
                       void **args)
    int GpuKernel_call2(_GpuKernel *, size_t n[2], size_t ls[2], size_t gs[2],
                        void **args)
    int GpuKernel_binary(_GpuKernel *, size_t *, void **)

cdef extern from "gpuarray/array.h":
    ctypedef struct _GpuArray "GpuArray":
        gpudata *data
        const gpuarray_buffer_ops *ops
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

    int GpuArray_empty(_GpuArray *a, const gpuarray_buffer_ops *ops, void *ctx,
                       int typecode, int nd, const size_t *dims, ga_order ord)
    int GpuArray_fromdata(_GpuArray *a, const gpuarray_buffer_ops *ops,
                          gpudata *data, size_t offset, int typecode,
                          unsigned int nd, const size_t *dims,
                          const ssize_t *strides, int writable)
    int GpuArray_copy_from_host(_GpuArray *a, const gpuarray_buffer_ops *ops,
                            void *ctx, void *buf, int typecode,
                            unsigned int nd, const size_t *dims,
                            const ssize_t *strides)
    int GpuArray_view(_GpuArray *v, _GpuArray *a)
    int GpuArray_sync(_GpuArray *a)
    int GpuArray_index(_GpuArray *r, _GpuArray *a, const ssize_t *starts,
                       const ssize_t *stops, const ssize_t *steps)
    int GpuArray_setarray(_GpuArray *v, _GpuArray *a)
    int GpuArray_reshape(_GpuArray *res, _GpuArray *a, unsigned int nd,
                         const size_t *newdims, ga_order ord, int nocopy)
    int GpuArray_transpose(_GpuArray *res, _GpuArray *a,
                           const unsigned int *new_axes)

    void GpuArray_clear(_GpuArray *a)

    int GpuArray_share(_GpuArray *a, _GpuArray *b)
    void *GpuArray_context(_GpuArray *a)

    int GpuArray_move(_GpuArray *dst, _GpuArray *src)
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz)
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src)
    int GpuArray_memset(_GpuArray *a, int data)
    int GpuArray_copy(_GpuArray *res, _GpuArray *a, ga_order order)

    int GpuArray_transfer(_GpuArray *res, const _GpuArray *a, void *new_ctx,
                          const gpuarray_buffer_ops *new_ops, int may_share)
    int GpuArray_split(_GpuArray **rs, const _GpuArray *a, size_t n,
                       size_t *p, unsigned int axis)
    int GpuArray_concatenate(_GpuArray *r, const _GpuArray **as, size_t n,
                             unsigned int axis, int restype)

    char *GpuArray_error(_GpuArray *a, int err)

    void GpuArray_fprintf(libc.stdio.FILE *fd, _GpuArray *a)
    bint GpuArray_is_c_contiguous(_GpuArray *a)
    bint GpuArray_is_f_contiguous(_GpuArray *a)

cdef extern from "gpuarray/extension.h":
    void *gpuarray_get_extension(const char *) nogil
    cdef int GPUARRAY_CUDA_CTX_NOFREE

# If you change the api interface, you MUST increment either the minor
# (if you add a function) or the major version (if you change
# arguments or remove a function) in the gpuarray.pyx file.
cdef api np.dtype typecode_to_dtype(int typecode)
cdef api int get_typecode(dtype) except -1
cpdef int dtype_to_typecode(dtype) except -1

cdef ga_order to_ga_order(ord) except <ga_order>-2

cdef bint py_CHKFLAGS(GpuArray a, int flags)
cdef bint py_ISONESEGMENT(GpuArray a)

cdef int array_empty(GpuArray a, const gpuarray_buffer_ops *ops, void *ctx,
                     int typecode, unsigned int nd, const size_t *dims,
                     ga_order ord) except -1
cdef int array_fromdata(GpuArray a, const gpuarray_buffer_ops *ops,
                        gpudata *data, size_t offset, int typecode,
                        unsigned int nd, const size_t *dims,
                        const ssize_t *strides, int writeable) except -1
cdef int array_copy_from_host(GpuArray a, const gpuarray_buffer_ops *ops,
                              void *ctx, void *buf, int typecode,
                              unsigned int nd, const size_t *dims,
                              const ssize_t *strides) except -1
cdef int array_view(GpuArray v, GpuArray a) except -1
cdef int array_sync(GpuArray a) except -1
cdef int array_index(GpuArray r, GpuArray a, const ssize_t *starts,
                     const ssize_t *stops, const ssize_t *steps) except -1
cdef int array_setarray(GpuArray v, GpuArray a) except -1
cdef int array_reshape(GpuArray res, GpuArray a, unsigned int nd,
                       const size_t *newdims, ga_order ord,
                       bint nocopy) except -1
cdef int array_transpose(GpuArray res, GpuArray a,
                         const unsigned int *new_axes) except -1
cdef int array_clear(GpuArray a) except -1
cdef bint array_share(GpuArray a, GpuArray b)
cdef void *array_context(GpuArray a) except NULL
cdef int array_move(GpuArray a, GpuArray src) except -1
cdef int array_write(GpuArray a, void *src, size_t sz) except -1
cdef int array_read(void *dst, size_t sz, GpuArray src) except -1
cdef int array_memset(GpuArray a, int data) except -1
cdef int array_copy(GpuArray res, GpuArray a, ga_order order) except -1
cdef int array_transfer(GpuArray res, GpuArray a, void *new_ctx,
                        const gpuarray_buffer_ops *ops,
                        bint may_share) except -1

cdef const char *kernel_error(GpuKernel k, int err) except NULL
cdef int kernel_init(GpuKernel k, const gpuarray_buffer_ops *ops, void *ctx,
                     unsigned int count, const char **strs, const size_t *len,
                     const char *name, unsigned int argcount, const int *types,
                     int flags) except -1
cdef int kernel_clear(GpuKernel k) except -1
cdef void *kernel_context(GpuKernel k) except NULL
cdef int kernel_call(GpuKernel k, size_t n, size_t ls, size_t gs,
                     void **args) except -1
cdef int kernel_call2(GpuKernel k, size_t n[2], size_t ls[2], size_t gs[2],
                     void **args) except -1
cdef int kernel_binary(GpuKernel k, size_t *, void **) except -1
cdef int kernel_property(GpuKernel k, int prop_id, void *res) except -1

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1
cdef const gpuarray_buffer_ops *get_ops(kind) except NULL
cdef ops_kind(const gpuarray_buffer_ops *ops)
cdef GpuContext ensure_context(GpuContext c)

cdef api GpuContext pygpu_default_context()

cdef api bint pygpu_GpuArray_Check(object o)

cdef api GpuContext pygpu_init(object dev)

cdef api GpuArray pygpu_zeros(unsigned int nd, const size_t *dims,
                              int typecode, ga_order order,
                              GpuContext context, type cls)
cdef api GpuArray pygpu_empty(unsigned int nd, const size_t *dims,
                              int typecode, ga_order order,
                              GpuContext context, type cls)
cdef api GpuArray pygpu_fromhostdata(void *buf, int typecode, unsigned int nd,
                                     const size_t *dims,
                                     const ssize_t *strides,
                                     GpuContext context, type cls)

cdef api GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
                                    unsigned int nd, const size_t *dims,
                                    const ssize_t *strides, GpuContext context,
                                    bint writable, object base, type cls)

cdef api GpuArray pygpu_copy(GpuArray a, ga_order ord)

cdef api int pygpu_move(GpuArray a, GpuArray src) except -1

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

cdef api GpuArray pygpu_transfer(GpuArray a, GpuContext new_ctx,
                                 bint may_share)
cdef api GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
                                    unsigned int axis, int restype,
                                    type cls, GpuContext context)

cdef api class GpuContext [type PyGpuContextType, object PyGpuContextObject]:
    cdef const gpuarray_buffer_ops *ops
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
    cdef void **callbuf
    cdef object __weakref__

    cdef do_call(self, py_n, py_ls, py_gs, py_args)
    cdef _setarg(self, unsigned int index, int typecode, object o)
