#include "private.h"

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

int gpucomm_new(gpucomm** comm, gpucontext* ctx,
                gpucommCliqueId cliqueId, int ndev, int rank) {
  if (ctx->comm_ops == NULL) {
    *comm = NULL;
    return GA_UNSUPPORTED_ERROR;
  }
  return ctx->comm_ops->comm_new(comm, ctx, cliqueId, ndev, rank);
}

int gpucomm_free(gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  ctx->comm_ops->comm_free(comm);
  return GA_NO_ERROR;
}

const char* gpucomm_error(gpucontext* ctx) {
  if (ctx->comm_ops != NULL)
      return ctx->comm_ops->comm_error(ctx);
  return "No collective ops available, API error. Is a collectives library installed?";
}

gpucontext* gpucomm_context(gpucomm* comm) {
  return ((partial_gpucomm*)comm)->ctx;
}

int gpucomm_gen_clique_id(gpucontext* ctx, gpucommCliqueId* cliqueId) {
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->generate_clique_id(ctx, cliqueId);
}

int gpucomm_get_count(gpucomm* comm, int* count) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->get_count(comm, count);
}

int gpucomm_get_device(gpucomm* comm, int* device) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->get_device(comm, device);
}

int gpucomm_get_rank(gpucomm* comm, int* rank) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->get_rank(comm, rank);
}

// Should we add some checks to ensure that gpudata src and dest sizes suffice
// for each op?
// e.g. for reduce: that (src->sz - offsrc) > count * sizeof(typecode)
// and (dest->sz - offdest) > count * sizeof(typecode)
// mby this should be done in gpuarray_buffer_collectives_cuda.c where scope of
// gpudata definition reaches.
int gpucomm_reduce(const gpudata* src, size_t offsrc,
                   gpudata* dest, size_t offdest,
                   int count, int typecode, int opcode,
                   int root, gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->reduce(src, offsrc, dest, offdest,
                                     count, typecode, opcode, root, comm);
}

int gpucomm_all_reduce(const gpudata* src, size_t offsrc,
                       gpudata* dest, size_t offdest,
                       int count, int typecode, int opcode,
                       gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->all_reduce(src, offsrc, dest, offdest,
                                         count, typecode, opcode, comm);
}

int gpucomm_reduce_scatter(const gpudata* src, size_t offsrc,
                           gpudata* dest, size_t offdest,
                           int count, int typecode, int opcode,
                           gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->reduce_scatter(src, offsrc, dest, offdest,
                                             count, typecode, opcode, comm);
}

int gpucomm_broadcast(gpudata* array, size_t offset,
                      int count, int typecode,
                      int root, gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->broadcast(array, offset,
                                        count, typecode, root, comm);
}

int gpucomm_all_gather(const gpudata* src, size_t offsrc,
                       gpudata* dest, size_t offdest,
                       int count, int typecode,
                       gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->all_gather(src, offsrc, dest, offdest,
                                         count, typecode, comm);
}
