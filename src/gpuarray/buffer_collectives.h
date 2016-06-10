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

enum _gpucomm_reduce_ops {
  GA_SUM = 0,
  GA_PROD = 1,
  GA_MAX = 2,
  GA_MIN = 3
};

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

/**
 * \brief TODO
 * \param res [int*] TODO
 * \return const char* TODO
 */
GPUARRAY_PUBLIC const char* gpucomm_gen_clique_id(int* res);

/**
 * \brief TODO
 * \param comm [const gpucomm*] TODO
 * \param count [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_count(const gpucomm* comm, int* count);

/**
 * \brief TODO
 * \param comm [const gpucomm*] TODO
 * \param device [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_device(const gpucomm* comm, int* device);

/**
 * \brief TODO
 * \param comm [const gpucomm*] TODO
 * \param rank [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_rank(const gpucomm* comm, int* rank);

// TODO private and cuda_private
// TODO cuda impl:
// functions in buffer, cuda buffer and cuda array
// adapter for typecode and opcode to nccl enums

// TODO redeclare below to add gpudata array metadata
/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param offsrc [size_t] TODO
 * \param dest [gpudata*] TODO
 * \param offdest [size_t] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_reduce(const gpudata* src, size_t offsrc,
                                   gpudata* dest, size_t offdest,
                                   int count, int typecode, int opcode,
                                   int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param offsrc [size_t] TODO
 * \param dest [gpudata*] TODO
 * \param offdest [size_t] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_all_reduce(const gpudata* src, size_t offsrc,
                                       gpudata* dest, size_t offdest,
                                       int count, int typecode, int opcode,
                                       gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param offsrc [size_t] TODO
 * \param dest [gpudata*] TODO
 * \param offdest [size_t] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param opcode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_reduce_scatter(const gpudata* src, size_t offsrc,
                                           gpudata* dest, size_t offdest,
                                           int count, int typecode, int opcode,
                                           gpucomm* comm);

/**
 * \brief TODO
 * \param array [gpudata*] TODO
 * \param offset [size_t] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param root [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_broadcast(gpudata* array, size_t offset,
                                      int count, int typecode,
                                      int root, gpucomm* comm);

/**
 * \brief TODO
 * \param src [const gpudata*] TODO
 * \param offsrc [size_t] TODO
 * \param dest [gpudata*] TODO
 * \param offdest [size_t] TODO
 * \param count [int] TODO
 * \param typecode [int] TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_all_gather(const gpudata* src, size_t offsrc,
                                       gpudata* dest, size_t offdest,
                                       int count, int typecode,
                                       gpucomm* comm);

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_BUFFER_COLLECTIVES_H
