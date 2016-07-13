from pygpu.gpuarray cimport (gpucontext, GpuContext, _GpuArray, GpuArray)

cdef extern from "gpuarray/buffer_collectives.h":
    ctypedef struct gpucomm:
        pass

    enum _gpucomm_reduce_ops:
        GA_SUM,
        GA_PROD,
        GA_MAX,
        GA_MIN

    enum: GA_COMM_ID_BYTES

    ctypedef struct gpucommCliqueId:
        char[GA_COMM_ID_BYTES] internal

    int gpucomm_new(gpucomm** comm, gpucontext* ctx,
                    gpucommCliqueId comm_id, int ndev, int rank)
    void gpucomm_free(gpucomm* comm)
    gpucontext* gpucomm_context(gpucomm* comm)
    int gpucomm_gen_clique_id(gpucontext* ctx, gpucommCliqueId* comm_id)
    int gpucomm_get_count(gpucomm* comm, int* gpucount)
    int gpucomm_get_rank(gpucomm* comm, int* rank)

cdef extern from "gpuarray/collectives.h" nogil:
    int GpuArray_reduce_from(const _GpuArray* src, int opcode,
                             int root, gpucomm* comm)
    int GpuArray_reduce(const _GpuArray* src, _GpuArray* dest,
                        int opcode, int root, gpucomm* comm)
    int GpuArray_all_reduce(const _GpuArray* src, _GpuArray* dest,
                            int opcode, gpucomm* comm)
    int GpuArray_reduce_scatter(const _GpuArray* src, _GpuArray* dest,
                                int opcode, gpucomm* comm)
    int GpuArray_broadcast(_GpuArray* array, int root, gpucomm* comm)
    int GpuArray_all_gather(const _GpuArray* src, _GpuArray* dest, gpucomm* comm)

cdef api class GpuCommCliqueId [type PyGpuCliqueIdType, object PyGpuCliqueIdObject]:
    cdef gpucommCliqueId c_comm_id
    cdef readonly GpuContext context


cdef api class GpuComm [type PyGpuCommType, object PyGpuCommObject]:
    cdef gpucomm* c
    cdef object __weakref__


cdef int to_reduce_opcode(op) except -1

cdef gpucontext* comm_context(GpuComm comm) except NULL
cdef int comm_generate_id(gpucontext* ctx, gpucommCliqueId* comm_id) except -1
cdef int comm_get_count(GpuComm comm, int* gpucount) except -1
cdef int comm_get_rank(GpuComm comm, int* gpurank) except -1
cdef int comm_reduce_from(GpuComm comm, GpuArray src, int opcode,
                          int root) except -1
cdef int comm_reduce(GpuComm comm, GpuArray src, GpuArray dest, int opcode,
                     int root) except -1
cdef int comm_all_reduce(GpuComm comm, GpuArray src, GpuArray dest,
                         int opcode) except -1
cdef int comm_reduce_scatter(GpuComm comm, GpuArray src, GpuArray dest,
                             int opcode) except -1
cdef int comm_broadcast(GpuComm comm, GpuArray arr, int root) except -1
cdef int comm_all_gather(GpuComm comm, GpuArray src, GpuArray dest) except -1

cdef api:
    GpuArray pygpu_make_reduced(GpuComm comm, GpuArray src, int opcode)
    GpuArray pygpu_make_all_reduced(GpuComm comm, GpuArray src, int opcode)
    GpuArray pygpu_make_reduce_scattered(GpuComm comm, GpuArray src, int opcode)
    GpuArray pygpu_make_all_gathered(GpuComm comm, GpuArray src,
                                     unsigned int nd_up)
