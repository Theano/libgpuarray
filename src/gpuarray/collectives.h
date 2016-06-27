#ifndef GPUARRAY_COLLECTIVES_H
#define GPUARRAY_COLLECTIVES_H

#include <gpuarray/array.h>
#include <gpuarray/buffer_collectives.h>
#include <gpuarray/config.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef CONFUSE_EMACS
}
#endif  // CONFUSE_EMACS

/*******************************************************************************
*                       Multi-gpu collectives interface                       *
*******************************************************************************/

/**
 * \brief Reduce collective operation for non root participant ranks in a
 * communicator world.
 * \param src [const GpuArray*] array to be reduced
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param root [int] rank in `comm` which will collect result
 * \param comm [gpucomm*] gpu communicator
 * \note Root rank of reduce operation must call \ref GpuArray_reduce.
 * \note Must be called separately for each rank in `comm`, except root rank.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_reduce_from(const GpuArray* src, int opcode,
                                         int root, gpucomm* comm);

/**
 * \brief Reduce collective operation for ranks in a communicator world.
 * \param src [const GpuArray*] array to be reduced
 * \param dest [GpuArray*] array to collect reduce operation result
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param root [int] rank in `comm` which will collect result
 * \param comm [gpucomm*] gpu communicator
 * \note Can be used by root and non root ranks alike.
 * \note Non root ranks can call this, using a NULL `dest`.
 * \note Must be called separately for each rank in `comm` (non root can call
 * \ref
 * GpuArray_reduce_from instead).
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_reduce(const GpuArray* src, GpuArray* dest,
                                    int opcode, int root, gpucomm* comm);

/**
 * \brief AllReduce collective operation for ranks in a communicator world.
 *
 * Reduces `src` using op operation and leaves identical copies of result in
 * `dest`
 * on each rank of `comm`.
 *
 * \param src [const GpuArray*] array to be reduced
 * \param dest [GpuArray*] array to collect reduce operation result
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_all_reduce(const GpuArray* src, GpuArray* dest,
                                        int opcode, gpucomm* comm);

/**
 * \brief ReduceScatter collective operation for ranks in a communicator world.
 *
 * Reduces data in `src` using `opcode` operation and leaves reduced result
 * scattered
 * over `dest` in the user-defined rank order in `comm`.
 *
 * \param src [const GpuArray*] array to be reduced
 * \param dest [GpuArray*] array to collect reduce operation scattered result
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_reduce_scatter(const GpuArray* src, GpuArray* dest,
                                            int opcode, gpucomm* comm);

/**
 * \brief Broadcast collective operation for ranks in a communicator world.
 *
 * Copies `array` to all ranks in `comm`.
 *
 * \param array [GpuArray*] array to be broadcasted, if root rank, else to
 * receive
 * \param root [int] rank in `comm` which broadcasts its array
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_broadcast(GpuArray* array, int root,
                                       gpucomm* comm);

/**
 * \brief AllGather collective operation for ranks in a communicator world.
 *
 * Each rank receives all `src` arrays from every rank in the user-defined rank
 * order
 * in `comm`.
 *
 * \param src [const GpuArray*] array to be gathered
 * \param dest [GpuArray*] array to receive all gathered arrays from ranks in
 * `comm`
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int GpuArray_all_gather(const GpuArray* src, GpuArray* dest,
                                        gpucomm* comm);

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_COLLECTIVES_H
