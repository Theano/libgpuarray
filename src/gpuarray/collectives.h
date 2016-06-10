#ifndef GPUARRAY_COLLECTIVES_H
#define GPUARRAY_COLLECTIVES_H

#include <gpuarray/config.h>
#include <gpuarray/array.h>
#include <gpuarray/buffer_collectives.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef CONFUSE_EMACS
}
#endif  // CONFUSE_EMACS

/************************************************************************************
*                         multi-gpu collectives interface                          *
************************************************************************************/

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param dest [GpuArray*] TODO
 * \param opcode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_reduce(const GpuArray* src, GpuArray* dest,
                                    int opcode, int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param dest [GpuArray*] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_all_reduce(const GpuArray* src, GpuArray* dest,
                                        int opcode, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param dest [GpuArray*] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_reduce_scatter(const GpuArray* src, GpuArray* dest,
                                            int opcode, gpucomm* comm);

/**
 * \brief TODO
 * \param array [GpuArray*] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_broadcast(GpuArray* array, int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param dest [GpuArray*] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_all_gather(const GpuArray* src, GpuArray* dest,
                                        gpucomm* comm);

// TODO discuss GpuArray layout restrictions on collectives
// 1. Reduce/AllReduce: consider as elemwise and check only total elem count? or
//                      consider to be consistent in dim size and order too?
// 2. Broadcast should probably consider consisent GpuArrays in everything.
// 3. ReduceScatter: at least a restriction for 'total elems of src' = ndev * 'total
// elems of dest'
// 4. AllGather: at least a restriction for 'total elems of dest' = ndev * 'total
// elems of src'

// TODO substitute int opcode with an enum opcode??

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_COLLECTIVES_H
