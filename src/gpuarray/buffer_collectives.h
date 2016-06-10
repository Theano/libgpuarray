#ifndef GPUARRAY_BUFFER_COLLECTIVES_H
#define GPUARRAY_BUFFER_COLLECTIVES_H

#include <gpuarray/config.h>
#include <gpuarray/buffer.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef CONFUSE_EMACS
}
#endif  // CONFUSE_EMACS

/************************************************************************************
*                      multi-gpu collectives buffer interface                      *
************************************************************************************/

/**
 * Multi-gpu communicator structure.
 *
 * \note The contents are private.
 */
struct _gpucomm;

typedef struct _gpucomm gpucomm;

// TODO add enum of gpucomm_reduce_ops

/**
 * \brief TODO
 * \param ctx [gpucontext*] TODO
 * \param clique_id [const char*] TODO
 * \param ndev [int] TODO
 * \param rank [int] TODO
 * \param res [int*] TODO
 * \return gpucomm* TODO
 */
GPUARRAY_PUBLIC gpucomm* gpucomm_new(gpucontext* ctx, const char* clique_id,
                                     int ndev, int rank, int* res);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \return void TODO
 */
GPUARRAY_PUBLIC void gpucomm_free(gpucomm* comm);


/**
 * \brief Returns nice error message.
 * \param comm [gpucomm*] TODO
 * \param err [int] TODO
 * \return const char* TODO
 */
GPUARRAY_PUBLIC const char* gpucomm_error(gpucomm* comm, int err);
// Mby check if err is GA_COMM_ERROR and if it is then call
// ctx->comm_ops->comm_error(ctx) else call gpucontext_error(ctx, err)
// same for gpublas for GA_BLAS_ERROR??

// TODO gpucomm
// 1. ncclGetUniqueId (bla bla)
// 2. ncclCommCount (exposed through comm_ops to gpucomm)
// 3. ncclCommUserRank (exposed though comm_ops to gpucomm)
// 4. ncclCommCuDevice (propably not in comm_ops)

// TODO private and cuda_private
// TODO cuda impl:
// functions in buffer, cuda buffer and cuda array
// adapter for typecode and opcode to nccl enums

// TODO redeclare below to add gpudata array metadata
/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param dest [gpudata*] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_reduce(const gpudata* src, gpudata* dest,
                                    int count, int typecode, int opcode,
                                    int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param dest [gpudata*] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_all_reduce(const gpudata* src, gpudata* dest,
                                        int count, int typecode, int opcode,
                                        gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param dest [gpudata*] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_reduce_scatter(const gpudata* src, gpudata* dest,
                                            int count, int typecode, int opcode,
                                            gpucomm* comm);

/**
 * \brief TODO
 * \param array [gpudata*] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_broadcast(gpudata* array, int count, int typecode,
                                       int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param dest [gpudata*] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_all_gather(const gpudata* src, gpudata* dest,
                                        int count, int typecode,
                                        gpucomm* comm);

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_BUFFER_COLLECTIVES_H
