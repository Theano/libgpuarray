#include <assert.h>
#include <stdlib.h>

#include <nccl.h>

#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"
#include "gpuarray/config.h"

#include "private.h"
#include "private_cuda.h"

#define NCCL_CHKFAIL(c)  { if ((c)->nccl_err != ncclSuccess) return GA_NO_ERROR; \
                           return GA_COMM_ERROR; }


extern const gpuarray_buffer_ops cuda_ops;

void comm_clear(gpucomm* comm)
{
  cuda_ops.buffer_deinit((gpucontext*) comm->ctx);
  CLEAR(comm);
  free(comm);
}

static int comm_new(gpucomm** comm_ptr, gpucontext* ctx, gpucommCliqueId comm_id,
                    int ndev, int rank) {
  ASSERT_CTX(ctx);
  gpucomm* comm;
  comm = calloc(1, sizeof(*comm));
  if (comm == NULL) {
    *comm_ptr = NULL;
    return GA_MEMORY_ERROR;
  }
  comm->ctx = (cuda_context*) ctx;
  comm->ctx->refcnt++;  // So that ctx would not be destroyed before comm
  cuda_enter(comm->ctx);
  comm->ctx->nccl_err = ncclCommInitRank(&comm->c, ndev,
                                         *((ncclUniqueId*)&comm_id), rank);
  cuda_exit(comm->ctx);
  TAG_COMM(comm);
  if (comm->ctx->nccl_err != ncclSuccess) {
    *comm_ptr = NULL;
    comm_clear(comm);
    return GA_COMM_ERROR;
  }
  *comm_ptr = comm;
  return GA_NO_ERROR;
}

static void comm_free(gpucomm* comm) {
  ASSERT_COMM(comm);
  cuda_enter(comm->ctx);
  ncclCommDestroy(comm->c);
  cuda_exit(comm->ctx);
  comm_clear(comm);
}

static const char* comm_error(gpucontext* c) {
  cuda_context* ctx = (cuda_context*) c;
  /* return "(nccl) " ncclGetErrorString(ctx->nccl_err); */
  switch (ctx->nccl_err) {
  case ncclSuccess                : return "(nccl) no error";
  case ncclUnhandledCudaError     : return "(nccl) unhandled cuda error";
  case ncclSystemError            : return "(nccl) system error";
  case ncclInternalError          : return "(nccl) internal error";
  case ncclInvalidDevicePointer   : return "(nccl) invalid device pointer";
  case ncclInvalidRank            : return "(nccl) invalid rank";
  case ncclUnsupportedDeviceCount : return "(nccl) unsupported device count";
  case ncclDeviceNotFound         : return "(nccl) device not found";
  case ncclInvalidDeviceIndex     : return "(nccl) invalid device index";
  case ncclLibWrapperNotSet       : return "(nccl) lib wrapper not initialized";
  case ncclCudaMallocFailed       : return "(nccl) cuda malloc failed";
  case ncclRankMismatch           : return "(nccl) parameter mismatch between ranks";
  case ncclInvalidArgument        : return "(nccl) invalid argument";
  case ncclInvalidType            : return "(nccl) invalid data type";
  case ncclInvalidOperation       : return "(nccl) invalid reduction operations";
  }
  return "(nccl) unknown result code";
}

static int generate_clique_id(gpucontext* c, gpucommCliqueId* comm_id) {
  cuda_context* ctx = (cuda_context*) c;
  ctx->nccl_err = ncclGetUniqueId((ncclUniqueId*) comm_id);
  NCCL_CHKFAIL(ctx);
}

static int get_count(const gpucomm* comm, int* count) {
  comm->ctx->nccl_err = ncclCommCount(comm->c, count);
  NCCL_CHKFAIL(comm->ctx);
}

static int get_rank(const gpucomm* comm, int* rank) {
  comm->ctx->nccl_err = ncclCommUserRank(comm->c, rank);
  NCCL_CHKFAIL(comm->ctx);
}

inline ncclRedOp_t convert_reduce_op(int opcode) {
  switch(opcode) {
  case GA_SUM : return ncclSum;
  case GA_PROD: return ncclProd;
  case GA_MAX : return ncclMax;
  case GA_MIN : return ncclMin;
  }
  return nccl_NUM_OPS;
}

inline ncclDataType_t convert_data_type(int typecode) {
  switch(typecode) {
  case GA_BYTE  : return ncclChar;
  case GA_INT   : return ncclInt;
#ifdef CUDA_HAS_HALF
  case GA_HALF  : return ncclHalf;
#endif  // CUDA_HAS_HALF
  case GA_FLOAT : return ncclFloat;
  case GA_DOUBLE: return ncclDouble;
  case GA_LONG  : return ncclInt64;
  case GA_ULONG : return ncclUint64;
  }
  return nccl_NUM_TYPES;
}

static int reduce(const gpudata* src, size_t offsrc,
                  gpudata* dest, size_t offdest,
                  int count, int typecode, int opcode,
                  int root, gpucomm* comm) {
}

static int all_reduce(const gpudata* src, size_t offsrc,
                      gpudata* dest, size_t offdest,
                      int count, int typecode, int opcode,
                      gpucomm* comm) {
}

static int reduce_scatter(const gpudata* src, size_t offsrc,
                          gpudata* dest, size_t offdest,
                          int count, int typecode, int opcode,
                          gpucomm* comm) {
}

static int broadcast(gpudata* array, size_t offset,
                     int count, int typecode,
                     int root, gpucomm* comm) {
}

static int all_gather(const gpudata* src, size_t offsrc,
                      gpudata* dest, size_t offdest,
                      int count, int typecode,
                      gpucomm* comm) {
}

GPUARRAY_LOCAL gpuarray_comm_ops nccl_ops = {
  comm_new,
  comm_free,
  comm_error,
  generate_clique_id,
  get_count,
  get_rank,
  reduce,
  all_reduce,
  reduce_scatter,
  broadcast,
  all_gather
};
