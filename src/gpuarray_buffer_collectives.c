#include "private.h"

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

int gpucomm_new(gpucomm* comm, gpucontext* ctx, const char* clique_id,
                     int ndev, int rank, int* res) {
  comm = NULL;
  if (ctx->comm_ops == NULL)
    return GA_UNSUPPORTED_ERROR;
  return ctx->comm_ops->comm_new(comm, ctx, clique_id, ndev, rank, res);
}

void gpucomm_free(gpucomm* comm) {
  gpucomm_context(comm)->comm_ops->comm_free(comm);
}

const char* gpucomm_error(gpucomm* comm, int err) {
  if (err == GA_COMM_ERROR && comm != NULL)
    return gpucomm_context(comm)->comm_ops->comm_error(gpucomm_context(comm));
  else
    return gpucontext_error(gpucomm_context(comm), err);
}

gpucontext* gpucomm_context(gpucomm* comm) {
  return ((partial_gpucomm*)comm)->ctx;
}

const char* gpucomm_gen_clique_id(int* res) {
  return gpucomm_context(comm)->comm_ops->generate_clique_id(res);
}

int gpucomm_get_count(const gpucomm* comm, int* count) {
  return gpucomm_context(comm)->comm_ops->get_count(comm, count);
}

int gpucomm_get_device(const gpucomm* comm, int* device) {
  return gpucomm_context(comm)->comm_ops->get_device(comm, device);
}

int gpucomm_get_rank(const gpucomm* comm, int* rank) {
  return gpucomm_context(comm)->comm_ops->get_rank(comm, rank);
}

// Should we add some checks to ensure that gpudata src and dest sizes suffice
// for each op?
// e.g. for reduce: that (src->sz - offsrc) > count * sizeof(typecode)
// and (dest->sz - offdest) > count * sizeof(typecode)
// mby this should be done in gpuarray_buffer_collectives_cuda.c where scope of
// gpudata definition reaches.
// Also, should we rename this ...collectives...? I think I did not choose a proper
// name - it's too long :P
int gpucomm_reduce(const gpudata* src, size_t offsrc,
                   gpudata* dest, size_t offdest,
                   int count, int typecode, int opcode,
                   int root, gpucomm* comm) {
  return gpucomm_context(comm)->comm_ops->reduce(src, offsrc, dest, offdest,
                                     count, typecode, opcode, root, comm);
}

int gpucomm_all_reduce(const gpudata* src, size_t offsrc,
                       gpudata* dest, size_t offdest,
                       int count, int typecode, int opcode,
                       gpucomm* comm) {
  return gpucomm_context(comm)->comm_ops->all_reduce(src, offsrc, dest, offdest,
                                         count, typecode, opcode, comm);
}

int gpucomm_reduce_scatter(const gpudata* src, size_t offsrc,
                           gpudata* dest, size_t offdest,
                           int count, int typecode, int opcode,
                           gpucomm* comm) {
  return gpucomm_context(comm)->comm_ops->reduce_scatter(src, offsrc, dest, offdest,
                                             count, typecode, opcode, comm);
}

int gpucomm_broadcast(gpudata* array, size_t offset,
                      int count, int typecode,
                      int root, gpucomm* comm) {
  return gpucomm_context(comm)->comm_ops->broadcast(array, offset,
                                        count, typecode, root, comm);
}

int gpucomm_all_gather(const gpudata* src, size_t offsrc,
                       gpudata* dest, size_t offdest,
                       int count, int typecode,
                       gpucomm* comm) {
  return gpucomm_context(comm)->comm_ops->all_gather(src, offsrc, dest, offdest,
                                         count, typecode, comm);
}
