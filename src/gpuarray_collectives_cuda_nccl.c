#include <nccl.h>

#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"
#include "gpuarray/config.h"

#include "private.h"
#include "private_cuda.h"

extern const gpuarray_buffer_ops cuda_ops;

static int comm_new(gpucomm** comm_ptr, gpucontext* ctx, gpucommCliqueId comm_id,
                    int ndev, int rank) {
  gpucomm* comm = (gpucomm*) calloc(1, sizeof(gpucomm));
  if (comm == NULL)
    return GA_MEMORY_ERROR;
  *comm_ptr = comm;
  comm->ctx = (cuda_context*) ctx;
  comm->ctx->refcnt++;  // So that ctx would not be destroyed before comm
  comm->ctx->nccl_err = ncclCommInitRank(&comm->c, ndev,
                                         *((ncclUniqueId*)&comm_id), rank);
  if (comm->ctx->nccl_err != ncclSuccess)
    return GA_COMM_ERROR;
  return GA_NO_ERROR;
}

static void comm_free(gpucomm* comm) {
}

static const char* comm_error(gpucontext* ctx) {
}

static int generate_clique_id(gpucontext* ctx, gpucommCliqueId* cliqueId) {
}

static int get_count(const gpucomm* comm, int* count) {
}

static int get_device(const gpucomm* comm, int* device) {
}

static int get_rank(const gpucomm* comm, int* rank) {
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
  get_device,
  get_rank,
  reduce,
  all_reduce,
  reduce_scatter,
  broadcast,
  all_gather
};
