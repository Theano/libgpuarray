cimport numpy
ctypedef void* gpubuf

cdef extern from *:
    cdef void cifcuda "#ifdef WITH_CUDA //" ()
    cdef void cifopencl "#ifdef WITH_OPENCL //" ()
    cdef void cendif "#endif //" ()

cdef extern from "compyte_buffer.h":
    struct compyte_buffer_ops:
        gpubuf (*buffer_alloc)(void* ctx, size_t sz)
        void (*buffer_free)(gpubuf buf)
        int (*buffer_move)(gpubuf dst, size_t dst_offset,
                           gpubuf src, size_t src_offset, size_t sz)
        int (*buffer_read)(void* dst, gpubuf src, 
                           size_t src_offset, size_t sz)
        int (*buffer_write)(gpubuf dst, size_t dst_offset, 
                            void* src, size_t sz)
        int (*buffer_memset)(gpubuf dst, int data, size_t sz)
        char* (*buffer_error)()

    compyte_buffer_ops _cuda_ops "cuda_ops"
    compyte_buffer_ops _opencl_ops "opencl_ops"

class BufferOpError(Exception):
    pass

cdef class py_gpubuf:
    cdef gpubuf ptr
    cdef void (*destroy)(gpubuf buf)

    def __init__(self):
        raise TypeError("This class cannot be instantiated from Python")
    
    def __dealloc__(self):
        self.destroy(self.ptr)

cdef py_gpubuf wrap_gpubuf(gpubuf ptr, void (*destroy)(gpubuf buf)):
    cdef py_gpubuf res = py_gpubuf.__new__(py_gpubuf)
    res.ptr = ptr
    res.destroy = destroy
    return  res

cdef class Ops:
    cdef compyte_buffer_ops *ops

    def __init__(self):
        raise TypeError("This class cannot be instantiated from Python")

    def alloc(self, ctx, size_t sz):
        if ctx is not None:
            raise TypeError("ctx must be None for now")
        cdef gpubuf res = self.ops.buffer_alloc(NULL, sz)
        if res == NULL:
            raise MemoryError(self.ops.buffer_error())
        return wrap_gpubuf(res, self.ops.buffer_free)

    def move(self, py_gpubuf dst not None, size_t dst_offset,
             py_gpubuf src not None, size_t src_offset, size_t sz):
        cdef int err = self.ops.buffer_move(dst.ptr, dst_offset, 
                                            src.ptr, src_offset, sz)
        if err == -1:
            raise BufferOpError(self.ops.buffer_error())

    def read(self, numpy.ndarray dst not None, 
             py_gpubuf src not None, size_t src_offset, size_t sz):
        
        cdef int err = self.ops.buffer_read(dst.data, src.ptr, src_offset, sz)
        if err == -1:
            raise BufferOpError(self.ops.buffer_error())

    def write(self, py_gpubuf dst not None, size_t dst_offset,
              numpy.ndarray src not None, size_t sz):
        cdef int err = self.ops.buffer_write(dst.ptr, dst_offset, src.data, sz)
        if err == -1:
            raise BufferOpError(self.ops.buffer_error())
    
    def memset(self, py_gpubuf dst not None, int data, size_t sz):
        cdef int err = self.ops.buffer_memset(dst.ptr, data, sz)
        if err == -1:
            raise BufferOpError(self.ops.buffer_error())

cdef Ops make_op(compyte_buffer_ops *ops):
    cdef Ops res = Ops.__new__(Ops)
    res.ops = ops
    return res

cifcuda()
cuda_ops = make_op(&_cuda_ops)
cendif()
cifopencl()
opencl_ops = make_op(&_opencl_ops)
cendif()
