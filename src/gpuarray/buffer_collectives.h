#ifndef GPUARRAY_BUFFER_COLLECTIVES_H
#define GPUARRAY_BUFFER_COLLECTIVES_H

#include "gpuarray/config.h"
#include "gpuarray/buffer.h"

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

/**
 * Enum for reduce ops of gpucomm
 */
enum _gpucomm_reduce_ops {
  GA_SUM = 0,
  GA_PROD = 1,
  GA_MAX = 2,
  GA_MIN = 3,
};

#define GA_COMM_ID_BYTES 128  // sizeof(gpucommCliqueId)
/**
 * Dummy struct to define byte-array's length through a type
 */
typedef struct _gpucommCliqueId {
  char (*internal)[][GA_COMM_ID_BYTES];
} gpucommCliqueId;

/**
 * \brief TODO
 * \param comm [gpucomm**] TODO
 * \param ctx [gpucontext*] TODO
 * \param cliqueId [gpucommCliqueId] TODO
 * \param ndev [int] TODO
 * \param rank [int] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_new(gpucomm** comm, gpucontext* ctx,
                                gpucommCliqueId cliqueId, int ndev, int rank);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_free(gpucomm* comm);


/**
 * \brief Returns nice error message concerning collectives array and buffer API.
 * \param ctx [gpucontext*] TODO
 * \return const char* TODO
 */
GPUARRAY_PUBLIC const char* gpucomm_error(gpucontext* ctx);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \return gpucontext* TODO
 */
GPUARRAY_PUBLIC gpucontext* gpucomm_context(gpucomm* comm);

/**
 * \brief TODO
 * \param ctx [gpucontext*] TODO
 * \param cliqueId [gpucommCliqueId*]
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_gen_clique_id(gpucontext* ctx,
                                          gpucommCliqueId* cliqueId);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \param count [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_count(gpucomm* comm, int* count);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \param device [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_device(gpucomm* comm, int* device);

/**
 * \brief TODO
 * \param comm [gpucomm*] TODO
 * \param rank [int*] TODO
 * \return int TODO
 */
GPUARRAY_PUBLIC int gpucomm_get_rank(gpucomm* comm, int* rank);

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
