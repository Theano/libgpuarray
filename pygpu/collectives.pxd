#  from pygpu.gpuarray import GpuArrayException
#  from pygpu.gpuarray cimport (_GpuArray, GpuArray, GA_NO_ERROR, GpuArray_error,
#                               pygpu_copy, pygpu_empty, pygpu_zeros,
#                               GA_ANY_ORDER, GA_F_ORDER, GpuArray_ISONESEGMENT)

cdef extern from "gpuarray/buffer_collectives.h":
    ctypedef struct gpucomm:
        pass

    enum _gpucomm_reduce_ops:
        GA_SUM,
        GA_PROD,
        GA_MAX,
        GA_MIN

    cdef int GA_COMM_ID_BYTES

    ctypedef struct gpucommCliqueId:
        bytes[GA_COMM_ID_BYTES] internal

    int gpucomm_new(gpucomm** comm, gpucontext* ctx,
                    gpucommCliqueId comm_id, int ndev, int rank)
    void gpucomm_free(gpucomm* comm)
    gpucontext* gpucomm_context(gpucomm* comm)
    int gpucomm_gen_clique_id(gpucontext* ctx, gpucommCliqueId* comm_id)
    int gpucomm_get_count(gpucomm* comm, int* gpucount)
    int gpucomm_get_rank(gpucomm* comm, int* rank)

cdef extern from "gpuarray/collectives.h":
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

cdef api class GpuComm [type PyGpuCommType, object PyGpuCommObject]:
    cdef gpucomm* c
    cdef gpucommCliqueId comm_id
    cdef readonly GpuContext context
    # cdef object __weakref__

cdef int comm_new(GpuComm comm, gpucontext* ctx, gpucommCliqueId comm_id,
                  int ndev, int rank) except -1
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

cdef api void pygpu_reduce_from(GpuComm comm, GpuArray src, int opcode, int root)
cdef api GpuArray pygpu_reduce(GpuComm comm, GpuArray src, int opcode)
cdef api GpuArray pygpu_all_reduce(GpuComm comm, GpuArray src, int opcode)
cdef api GpuArray pygpu_reduce_scatter(GpuComm comm, GpuArray src, int opcode)
cdef api GpuArray pygpu_bcast_receive(GpuComm comm, int root)
cdef api void pygpu_bcast_send(GpuComm comm, GpuArray arr)
cdef api GpuArray pygpu_all_gather(GpuComm comm, GpuArray src)
