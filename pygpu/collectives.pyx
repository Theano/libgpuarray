cdef class GpuCommCliqueId:
    """
    """
    def __cinit__(self, GpuContext context=None):
        self.context = ensure_context(context)

    property comm_id:
        "Unique clique id to be used by each GpuComm in a group of devices"
        def __get__(self):
            comm_generate_id(self.context.ctx, self)
            return self.comm_id.internal  # cast to python byte array/string

        def __set__(self, bytearr):
            if len(bytearr) < GA_COMM_ID_BYTES:
                raise ValueError, "gpucomm clique id must have length " + str(GA_COMM_ID_BYTES) + " bytes"
            # Make sure that either reference is kept or copy
            # Cast to bytearray
            self.comm_id.internal = bytearr[:GA_COMM_ID_BYTES]


cdef class GpuComm:
    """
    """
    def __dealloc__(self):
        gpucomm_free(self.c)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuComm object"

    def __cinit__(self, GpuCommCliqueId cid, int ndev, int rank):
        self.context = cid.context
        cdef int err
        err = gpucomm_new(&self.c, self.context.ctx, self.cid.comm_id,
                          ndev, rank)
        if err != GA_NO_ERROR:
            raise get_exc(err), gpucontext_error(self.context.ctx, err)

    def get_count(self):
        cdef int gpucount
        comm_get_count(self, &gpucount)
        return gpucount

    def get_rank(self):
        cdef int gpurank
        comm_get_rank(self, &gpurank)
        return gpurank

    def reduce(self, GpuArray src, op, GpuArray dest=None, int root=None):
        if not dest:
            if root:
                return comm_reduce_from(self, src, to_reduce_opcode(op), root)
            else:
                return pygpu_make_reduced(self, src, to_reduce_opcode(op))
        if not root:
            comm_get_rank(self, &root)
        return comm_reduce(self, src, dest, to_reduce_opcode(op), root)

    def all_reduce(self, GpuArray src, op, GpuArray dest=None):
        if not dest:
            return pygpu_make_all_reduced(self, src, to_reduce_opcode(op))
        return comm_all_reduce(self, src, dest, to_reduce_opcode(op))

    def reduce_scatter(self, GpuArray src, op, GpuArray dest=None):
        if not dest:
            return pygpu_make_reduce_scattered(self, src, to_reduce_opcode(op))
        return comm_reduce_scatter(self, src, dest, to_reduce_opcode(op))

    def broadcast(self, Gpuarray array, int root=None):
        if not root:
            comm_get_rank(self, &root)
        return comm_broadcast(self, array, root)

    def all_gather(self, GpuArray src, GpuArray dest=None,
                   unsigned int nd_up=1):
        if not dest:
            return pygpu_make_all_gathered(self, src, nd_up)
        return comm_all_gather(self, src, dest)


cdef dict TO_RED_OP = {
    '+': GA_SUM,
    "sum": GA_SUM,
    "add": GA_SUM,
    '*': GA_PROD,
    "prod": GA_PROD,
    "product": GA_PROD,
    "max": GA_MAX,
    "maximum": GA_MAX,
    "min": GA_MIN,
    "minimum": GA_MIN,
    }

cdef int to_reduce_opcode(op):
    if isinstance(op, int):
        return op
    res = TO_RED_OP.get(op.lower())
    if res:
        return res
    raise ValueError, "Invalid reduce operation"

cdef gpucontext* comm_context(GpuComm comm) except NULL:
    cdef gpucontext* res
    res = gpucomm_context(comm.c)
    if res is NULL:
        raise GpuArrayException, "Invalid communicator or destroyed context"
    return res

cdef int comm_generate_id(gpucontext* ctx, GpuCommCliqueId comm_id) except -1:
    cdef int err
    err = gpucomm_gen_clique_id(ctx, &comm_id.comm_id)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int comm_get_count(GpuComm comm, int* gpucount) except -1:
    cdef int err
    err = gpucomm_get_count(comm.c, gpucount)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_get_rank(GpuComm comm, int* gpurank) except -1:
    cdef int err
    err = gpucomm_get_count(comm.c, gpurank)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_reduce_from(GpuComm comm, GpuArray src, int opcode,
                          int root) except -1:
    cdef int err
    err = GpuArray_reduce_from(&src.ga, opcode, root, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_reduce(GpuComm comm, GpuArray src, GpuArray dest, int opcode,
                     int root) except -1:
    cdef int err
    err = GpuArray_reduce(&src.ga, &dest.ga, opcode, root, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_all_reduce(GpuComm comm, GpuArray src, GpuArray dest,
                         int opcode) except -1:
    cdef int err
    err = GpuArray_all_reduce(&src.ga, &dest.ga, opcode, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_reduce_scatter(GpuComm comm, GpuArray src, GpuArray dest,
                             int opcode) except -1:
    cdef int err
    err = GpuArray_reduce_scatter(&src.ga, &dest.ga, opcode, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_broadcast(GpuComm comm, GpuArray arr, int root) except -1:
    cdef int err
    err = GpuArray_broadcast(&array.ga, root, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef int comm_all_gather(GpuComm comm, GpuArray src, GpuArray dest) except -1:
    cdef int err
    err = GpuArray_all_gather(&src.ga, &dest.ga, comm.c)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(comm_context(comm), err)

cdef api GpuArray pygpu_make_reduced(GpuComm comm, GpuArray src, int opcode):
    cdef GpuArray res
    res = pygpu_empty_like(src, GA_ANY_ORDER, -1)
    cdef int rank
    comm_get_rank(comm, &rank)
    comm_reduce(comm, src, res, opcode, rank)
    return res

cdef api GpuArray pygpu_make_all_reduced(GpuComm comm, GpuArray src, int opcode):
    cdef GpuArray res
    res = pygpu_empty_like(src, GA_ANY_ORDER, -1)
    comm_all_reduce(comm, src, res, opcode)
    return res

cdef api GpuArray pygpu_make_reduce_scattered(GpuComm comm, GpuArray src, int opcode):
    if src.ga.nd < 1:
        raise TypeError, "Source GpuArray must have number of dimensions >= 1"

    cdef GpuArray res
    cdef int gpucount
    cdef bint is_c_cont
    cdef unsigned int nd
    cdef size_t chosen_dim_size
    cdef size_t* dims

    comm_get_count(comm, &gpucount)
    is_c_cont = GpuArray_IS_C_CONTIGUOUS(&src.ga)
    nd = src.ga.nd
    dims = <size_t*>calloc(nd, sizeof(size_t))
    if dims == NULL:
        raise MemoryError, "Could not allocate dims"

    try:
        if is_c_cont:
            # Smallest in index dimension has the largest stride
            if src.ga.dimensions[0] % gpucount == 0:
                chosen_dim_size = src.ga.dimensions[0] / gpucount
                if chosen_dim_size != 1:
                    dims[0] = chosen_dim_size
                    for j in range(1, nd):
                        dims[j] = src.ga.dimensions[j]
                else:
                    for j in range(nd - 1):
                        dims[j] = src.ga.dimensions[1 + j]
                    nd -= 1
            else:
                raise TypeError, "Source GpuArray cannot be split in %d c-contiguous arrays" % (gpucount)
        else:
            # Largest in index dimension has the largest stride
            if src.ga.dimensions[nd - 1] % gpucount == 0:
                chosen_dim_size = src.ga.dimensions[nd - 1] / gpucount
                for j in range(nd - 1):
                    dims[j] = src.ga.dimensions[j]
                if chosen_dim_size != 1:
                    dims[nd - 1] = chosen_dim_size
                else:
                    nd -= 1
            else:
                raise TypeError, "Source GpuArray cannot be split in %d c-contiguous arrays" % (gpucount)
        res = pygpu_empty(nd, dims, src.ga.typecode,
                          GA_C_ORDER if is_c_cont else GA_F_ORDER,
                          src.context, type(src))
        comm_reduce_scatter(comm, src, res, opcode)
    finally:
        free(dims)

    return res

cdef api GpuArray pygpu_make_all_gathered(GpuComm comm, GpuArray src,
                                          unsigned int nd_up):
    if src.ga.nd < 1:
        raise TypeError, "Source GpuArray must have number of dimensions >= 1"

    cdef GpuArray res
    cdef int gpucount
    cdef bint is_c_cont
    cdef unsigned int nd
    cdef size_t* dims

    comm_get_count(comm, &gpucount)
    is_c_cont = GpuArray_IS_C_CONTIGUOUS(&src.ga)
    nd = src.ga.nd + nd_up
    dims = <size_t*>calloc(nd, sizeof(size_t))
    if dims == NULL:
        raise MemoryError, "Could not allocate dims"

    try:
        if is_c_cont:
            # Smallest in index dimension has the largest stride
            if nd_up == 0:
                dims[0] = <size_t>gpucount * src.ga.dimensions[0]
                for j in range(1, nd):
                    dims[j] = src.ga.dimensions[j]
            else:
                dims[0] = <size_t>gpucount
                for j in range(1, nd_up):
                    dims[j] = 1
                for j in range(src.ga.nd):
                    dims[nd_up + j] = src.ga.dimensions[j]
        else:
            # Largest in index dimension has the largest stride
            if nd_up == 0:
                dims[nd - 1] = <size_t>gpucount * src.ga.dimensions[nd - 1]
                for j in range(nd - 1):
                    dims[j] = src.ga.dimensions[j]
            else:
                dims[nd - 1] = <size_t>gpucount
                for j in range(nd_up - 1):
                    dims[src.ga.nd + j] = 1
                for j in range(src.ga.nd):
                    dims[j] = src.ga.dimensions[j]
        res = pygpu_empty(nd, dims, src.ga.typecode,
                          GA_C_ORDER if is_c_cont else GA_F_ORDER,
                          src.context, type(src))
        comm_all_gather(comm, src, res, opcode)
    finally:
        free(dims)

    return res
