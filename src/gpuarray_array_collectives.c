#include "gpuarray/array.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/collectives.h"
#include "gpuarray/error.h"

#include "private.h"

/**
 * \brief Finds total number of elements contained in `array`.
 */
static inline size_t find_total_elems(const GpuArray* array) {
  unsigned int i;
  size_t total_elems = 1;
  for (i = 0; i < array->nd; ++i)
    total_elems *= array->dimensions[i];
  return total_elems;
}

/**
 * \brief Checks if `src` and `dest` arrays are appropriate to participate in a
 * collective operation.
 *
 * Checks to see if they contain the appropriate number of elements, if they are
 * properly aligned (contiguous) and writeable (for `dest`) and if they contain
 * elements of the same datatype. It returns the number of elements of the array
 * with
 * the less length.
 */
static inline int check_gpuarrays(int times_src, const GpuArray* src,
                                  int times_dest, const GpuArray* dest,
                                  size_t* count) {
  gpucontext *ctx = gpudata_context(src->data);
  size_t count_src, count_dest;
  count_src = find_total_elems(src);
  count_dest = find_total_elems(dest);
  if (times_src * count_src != times_dest * count_dest)
    return error_set(ctx->err, GA_VALUE_ERROR, "Size mismatch for transfer");
  if (src->typecode != dest->typecode)
    return error_set(ctx->err, GA_VALUE_ERROR, "Type mismatch");
  if (!GpuArray_ISALIGNED(src) || !GpuArray_ISALIGNED(dest))
    return error_set(ctx->err, GA_UNALIGNED_ERROR, "Unaligned arrays");
  if (!GpuArray_ISWRITEABLE(dest))
    return error_set(ctx->err, GA_INVALID_ERROR, "Unwritable destination");

  if (times_src >= times_dest)
    *count = count_src;
  else
    *count = count_dest;
  return GA_NO_ERROR;
}

int GpuArray_reduce_from(const GpuArray* src, int opcode, int root,
                         gpucomm* comm) {
  gpucontext *ctx = gpudata_context(src->data);
  size_t total_elems;
  if (!GpuArray_ISALIGNED(src))
    return error_set(ctx->err, GA_UNALIGNED_ERROR, "Unaligned input");
  total_elems = find_total_elems(src);
  return gpucomm_reduce(src->data, src->offset, NULL, 0, total_elems,
                        src->typecode, opcode, root, comm);
}

int GpuArray_reduce(const GpuArray* src, GpuArray* dest, int opcode, int root,
                    gpucomm* comm) {
  int rank = 0;
  GA_CHECK(gpucomm_get_rank(comm, &rank));
  if (rank == root) {
    size_t count = 0;
    GA_CHECK(check_gpuarrays(1, src, 1, dest, &count));
    return gpucomm_reduce(src->data, src->offset, dest->data, dest->offset,
                          count, src->typecode, opcode, root, comm);
  } else {
    return GpuArray_reduce_from(src, opcode, root, comm);
  }
}

int GpuArray_all_reduce(const GpuArray* src, GpuArray* dest, int opcode,
                        gpucomm* comm) {
  size_t count = 0;
  GA_CHECK(check_gpuarrays(1, src, 1, dest, &count));
  return gpucomm_all_reduce(src->data, src->offset, dest->data, dest->offset,
                            count, src->typecode, opcode, comm);
}

int GpuArray_reduce_scatter(const GpuArray* src, GpuArray* dest, int opcode,
                            gpucomm* comm) {
  size_t count = 0;
  int ndev = 0;
  GA_CHECK(gpucomm_get_count(comm, &ndev));
  GA_CHECK(check_gpuarrays(1, src, ndev, dest, &count));
  return gpucomm_reduce_scatter(src->data, src->offset, dest->data,
                                dest->offset, count, src->typecode, opcode,
                                comm);
}

int GpuArray_broadcast(GpuArray *array, int root, gpucomm *comm) {
  gpucontext *ctx = gpudata_context(array->data);
  size_t total_elems;
  int rank = 0;

  GA_CHECK(gpucomm_get_rank(comm, &rank));
  if (rank == root) {
    if (!GpuArray_CHKFLAGS(array, GA_BEHAVED))
      return error_set(ctx->err, GA_UNALIGNED_ERROR, "Unaligned input");
  } else {
    if (!GpuArray_ISALIGNED(array))
      return error_set(ctx->err, GA_UNALIGNED_ERROR, "Unaligned input");
  }

  total_elems = find_total_elems(array);

  return gpucomm_broadcast(array->data, array->offset, total_elems,
                           array->typecode, root, comm);
}

int GpuArray_all_gather(const GpuArray* src, GpuArray* dest, gpucomm* comm) {
  size_t count = 0;
  int ndev = 0;
  GA_CHECK(gpucomm_get_count(comm, &ndev));
  GA_CHECK(check_gpuarrays(ndev, src, 1, dest, &count));
  return gpucomm_all_gather(src->data, src->offset, dest->data, dest->offset,
                            count, src->typecode, comm);
}
