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

/************************************************************************************
*                         multi-gpu collectives interface                          *
************************************************************************************/

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param opcode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_reduce_from(const GpuArray* src, int opcode, int root,
                                         gpucomm* comm);

/**
 * \brief TODO
 * \param src [const GpuArray*] TODO
 * \param dest [GpuArray*] TODO
 * \param opcode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int GpuArray_reduce(const GpuArray* src, GpuArray* dest, int opcode,
                                    int root, gpucomm* comm);

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

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_COLLECTIVES_H
