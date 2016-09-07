from pygpu.gpuarray import GpuArrayException
from pygpu.gpuarray cimport (gpucontext, GA_NO_ERROR, get_typecode,
                             typecode_to_dtype, GpuContext, GpuArray,
                             get_exc, gpuarray_get_elsize)
from pygpu.gpuarray cimport (GA_BUFFER, GA_SIZE, GA_SSIZE, GA_ULONG, GA_LONG,
                             GA_UINT, GA_INT, GA_USHORT, GA_SHORT,
                             GA_UBYTE, GA_BYTE, GA_DOUBLE, GA_FLOAT)
from libc.string cimport memset, memcpy, strdup
from libc.stdlib cimport malloc, calloc, free

cdef bytes to_bytes(s):
  if isinstance(s, bytes):
      return <bytes>s
  if isinstance(s, unicode):
      return <bytes>(<unicode>s).encode('ascii')
  raise TypeError("Can't convert to bytes")

cdef extern from "gpuarray/elemwise.h":
    ctypedef struct _GpuElemwise "GpuElemwise":
        pass

    ctypedef struct gpuelemwise_arg:
        const char *name
        int typecode
        int flags

    cdef int GE_SCALAR
    cdef int GE_READ
    cdef int GE_WRITE

    _GpuElemwise *GpuElemwise_new(gpucontext *ctx, const char *preamble,
                                  const char *expr, unsigned int n,
                                  gpuelemwise_arg *args, unsigned int nd,
                                  int flags)
    void GpuElemwise_free(_GpuElemwise *ge)
    int GpuElemwise_call(_GpuElemwise *ge, void **args, int flags)

    cdef int GE_NOADDR64
    cdef int GE_CONVERT_F16

    cdef int GE_BROADCAST
    cdef int GE_NOCOLLAPSE


cdef class arg:
    cdef gpuelemwise_arg a

    def __cinit__(self):
        memset(&self.a, 0, sizeof(gpuelemwise_arg))

    def __init__(self, name, type, read=False, write=False, scalar=False):
        self.a.name = strdup(to_bytes(name))
        if self.a.name is NULL:
            raise MemoryError
        self.a.typecode = get_typecode(type)
        self.a.flags = 0
        if read:
            self.a.flags |= GE_READ
        if write:
            self.a.flags |= GE_WRITE
        if scalar:
            self.a.flags |= GE_SCALAR
        if self.a.flags == 0:
            raise ValueError('no flags specified for arg %s' % (name,))

    property name:
        def __get__(self):
            return self.a.name.decode('ascii')

    property type:
        def __get__(self):
            return typecode_to_dtype(self.a.typecode)

    property read:
        def __get__(self):
            return self.a.flags & GE_READ

    property write:
        def __get__(self):
            return self.a.flags & GE_WRITE
    property scalar:
        def __get__(self):
            return self.a.flags & GE_SCALAR


cdef class GpuElemwise:
    cdef _GpuElemwise *ge
    cdef int *types
    cdef void **callbuf
    cdef unsigned int n

    def __cinit__(self, GpuContext ctx, expr, args, unsigned int nd=0,
                  preamble=b"", bint convert_f16=False):
        cdef gpuelemwise_arg *_args;
        cdef unsigned int i
        cdef arg aa

        self.ge = NULL
        self.types = NULL
        self.callbuf = NULL

        preamble = to_bytes(preamble)
        expr = to_bytes(expr)
        self.n = len(args)

        self.types = <int *>calloc(self.n, sizeof(int))
        if self.types is NULL:
            raise MemoryError

        self.callbuf = <void **>calloc(self.n, sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError

        _args = <gpuelemwise_arg *>calloc(self.n, sizeof(gpuelemwise_arg));
        if _args is NULL:
            raise MemoryError
        try:
            for i in range(self.n):
                if not isinstance(args[i], arg):
                    raise TypeError("args must be an iterable of arg")
                aa = <arg>args[i]
                memcpy(&_args[i], &aa.a, sizeof(gpuelemwise_arg))
                if aa.a.flags & GE_SCALAR:
                    self.types[i] = aa.a.typecode
                    self.callbuf[i] = malloc(gpuarray_get_elsize(aa.a.typecode))
                    if self.callbuf[i] is NULL:
                        raise MemoryError
                else:
                    self.types[i] = GA_BUFFER

            self.ge = GpuElemwise_new(ctx.ctx, preamble, expr, self.n,
                                      _args, nd,
                                      GE_CONVERT_F16 if convert_f16 else 0)
        finally:
            free(_args)
        if self.ge is NULL:
            raise GpuArrayException("Could not initialize C GpuElemwise instance")

    def __dealloc__(self):
        cdef unsigned int i

        if self.ge is not NULL:
            GpuElemwise_free(self.ge)
            self.ge = NULL
        for i in range(self.n):
            if self.types[i] != GA_BUFFER:
                free(self.callbuf[i])
        free(self.callbuf)
        free(self.types)

    cdef _setarg(self, unsigned int index, object o):
        cdef int typecode
        typecode = self.types[index]

        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
                raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>&(<GpuArray>o).ga
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

    def __call__(self, *args, **kwargs):
        cdef unsigned int i
        cdef int err

        for i, arg in enumerate(args):
            self._setarg(i, arg)
        err = GpuElemwise_call(self.ge, self.callbuf, GE_BROADCAST if kwargs.get('broadcast', True) else 0)
        if err != GA_NO_ERROR:
            raise get_exc(err)("Could not call GpuElemwise")
