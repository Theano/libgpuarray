#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

#include "private.h"

int gpucomm_new(gpucomm** comm, gpucontext* ctx, gpucommCliqueId comm_id,
                int ndev, int rank) {
  if (ctx->comm_ops == NULL) {
    *comm = NULL;
    return GA_UNSUPPORTED_ERROR;
  }
  return ctx->comm_ops->comm_new(comm, ctx, comm_id, ndev, rank);
}

void gpucomm_free(gpucomm* comm) {
  gpucontext* ctx;
  if (comm == NULL) return;
  ctx = gpucomm_context(comm);
  if (ctx->comm_ops != NULL)
    ctx->comm_ops->comm_free(comm);
}

const char* gpucomm_error(gpucontext* ctx) {
  if (ctx->comm_ops != NULL)
    return ctx->error_msg;
  return "No collective ops available, API error. Is a collectives library "
         "installed?";
}

gpucontext* gpucomm_context(gpucomm* comm) {
  return ((partial_gpucomm*)comm)->ctx;
}
int gpucomm_gen_clique_id(gpucontext* ctx, gpucommCliqueId* comm_id) {
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->generate_clique_id(ctx, comm_id);
}

int gpucomm_get_count(gpucomm* comm, int* gpucount) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->get_count(comm, gpucount);
}

int gpucomm_get_rank(gpucomm* comm, int* rank) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->get_rank(comm, rank);
}

int gpucomm_reduce(gpudata* src, size_t offsrc, gpudata* dest, size_t offdest,
                   size_t count, int typecode, int opcode, int root,
                   gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->reduce(src, offsrc, dest, offdest, count, typecode,
                               opcode, root, comm);
}

int gpucomm_all_reduce(gpudata* src, size_t offsrc, gpudata* dest,
                       size_t offdest, size_t count, int typecode, int opcode,
                       gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->all_reduce(src, offsrc, dest, offdest, count, typecode,
                                   opcode, comm);
}

int gpucomm_reduce_scatter(gpudata* src, size_t offsrc, gpudata* dest,
                           size_t offdest, size_t count, int typecode,
                           int opcode, gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->reduce_scatter(src, offsrc, dest, offdest, count,
                                       typecode, opcode, comm);
}

int gpucomm_broadcast(gpudata* array, size_t offset, size_t count, int typecode,
                      int root, gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->broadcast(array, offset, count, typecode, root, comm);
}

int gpucomm_all_gather(gpudata* src, size_t offsrc, gpudata* dest,
                       size_t offdest, size_t count, int typecode,
                       gpucomm* comm) {
  gpucontext* ctx = gpucomm_context(comm);
  if (ctx->comm_ops == NULL)
    return GA_COMM_ERROR;
  return ctx->comm_ops->all_gather(src, offsrc, dest, offdest, count, typecode,
                                   comm);
}
