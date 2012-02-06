cimport libc.stdio
from libc.stdlib cimport calloc, free
cimport numpy

cdef extern from "compyte_buffer.h":
    ctypedef struct gpudata:
        pass
    ctypedef struct gpukernel:
        pass

    ctypedef struct compyte_buffer_ops:
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

    ctypedef enum ga_order:
        GA_ANY_ORDER, GA_C_ORDER, GA_F_ORDER

    cdef enum ga_error:
        GA_NO_ERROR, GA_MEMORY_ERROR, GA_VALUE_ERROR, GA_IMPL_ERROR,
        GA_INVALID_ERROR, GA_UNSUPPORTED_ERROR, GA_SYS_ERROR

    cdef int GA_NBASE

    int GpuArray_empty(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)
    int GpuArray_zeros(_GpuArray *a, compyte_buffer_ops *ops, void *ctx,
                       int typecode, int nd, size_t *dims, ga_order ord)

    void GpuArray_clear(_GpuArray *a)
    
    int GpuArray_move(_GpuArray *dst, _GpuArray *src)
    int GpuArray_write(_GpuArray *dst, void *src, size_t src_sz)
    int GpuArray_read(void *dst, size_t dst_sz, _GpuArray *src)
    
    void GpuArray_fprintf(libc.stdio.FILE *fd, _GpuArray *a)
    int GpuArray_is_c_contiguous(_GpuArray *a)
    int GpuArray_is_f_contiguous(_GpuArray *a)

import numpy

class type:
    CUDA = 0
    OPENCL = 1

class order:
    ANY = GA_ANY_ORDER
    F = GA_F_ORDER
    C = GA_C_ORDER

cdef int dtype_to_typecode(dtype):
    cdef int dnum
    if isinstance(dtype, int):
        return dtype
    if isinstance(dtype, numpy.dtype):
        dnum = (<numpy.dtype>dtype).type_num
        if dnum < GA_NBASE:
            return dnum
    raise ValueError("don't know how to convert to dtype: %s"%(dtype,))

class GpuArrayException(Exception):
    pass

cdef class GpuArray:
    cdef _GpuArray ga

    def __cinit__(self, type, size_t ctx, dtype, dims, ga_order ord=order.ANY,
                  *a, **kwa):
        cdef compyte_buffer_ops *o
        cdef size_t *cdims
        cdef int err

        cdims = <size_t *>calloc(len(dims), sizeof(size_t));
        if cdims == NULL:
            raise MemoryError("could not allocate cdims")
        for i, d in enumerate(dims):
            cdims[i] = d
        
        if type == type.CUDA:
            o = &cuda_ops
        elif type == type.OPENCL:
            o = &opencl_ops
        else:
            raise ValueError("unknown type %d"%(type,))
        
        err = GpuArray_empty(&self.ga, o, <void *>ctx,
                             dtype_to_typecode(dtype),
                             len(dims), cdims, ord)
        free(cdims)
        
        if (err != GA_NO_ERROR):
            raise GpuArrayException(o.buffer_error())        

    def __dealloc__(self):
        GpuArray_clear(&self.ga)
