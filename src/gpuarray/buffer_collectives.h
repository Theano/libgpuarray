#ifndef GPUARRAY_BUFFER_COLLECTIVES_H
#define GPUARRAY_BUFFER_COLLECTIVES_H

#include <gpuarray/buffer.h>
#include <gpuarray/config.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
#ifdef CONFUSE_EMACS
}
#endif  // CONFUSE_EMACS

/*******************************************************************************
*                   Multi-gpu collectives buffer interface                    *
*******************************************************************************/

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
  GA_SUM = 0,   //!< to sum (elemwise) arrays across ranks
  GA_PROD = 1,  //!< to multiply (elemwise) arrays across ranks
  GA_MAX = 2,   //!< to find max (elemwise) of arrays across ranks
  GA_MIN = 3,   //!< to find min (elemwise) of arrays across ranks
};

#define GA_COMM_ID_BYTES 128  //!< sizeof(gpucommCliqueId)

/**
 * Dummy struct to define byte-array's length through a type
 */
typedef struct _gpucommCliqueId {
  char internal[GA_COMM_ID_BYTES];
} gpucommCliqueId;

/**
 * \brief Create a new gpu communicator instance.
 * \param comm [gpucomm**] pointer to get a new gpu communicator
 * \param ctx [gpucontext*] gpu context in which `comm` will be used (contains
 * device
 * information)
 * \param comm_id [gpucommCliqueId] id unique to communicators consisting a
 * world
 * \param ndev [int] number of communicators/devices participating in the world
 * \param rank [int] user-defined rank, from 0 to `ndev`-1, of `comm` in the
 * world
 * \note `rank` is defined to be unique for each new `comm` participating in the
 * same
 * world.
 * \note Must be called in parallel by all separate new `comm`, which will
 * consist a
 * new world (failing will lead to deadlock).
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_new(gpucomm** comm, gpucontext* ctx,
                                gpucommCliqueId comm_id, int ndev, int rank);

/**
 * \brief Destroy a gpu communicator instance.
 * \param comm [gpucomm*] gpu communicator to be destroyed
 * \return void
 */
GPUARRAY_PUBLIC void gpucomm_free(gpucomm* comm);

/**
 * \brief Returns nice error message concerning \ref GA_COMM_ERROR.
 * \param ctx [gpucontext*] gpu context in which communicator was used
 * \return const char* useful backend error message
 */
GPUARRAY_PUBLIC const char* gpucomm_error(gpucontext* ctx);

/**
 * \brief Returns gpu context in which `comm` is used.
 * \param comm [gpucomm*] gpu communicator
 * \return gpucontext* gpu context
 */
GPUARRAY_PUBLIC gpucontext* gpucomm_context(gpucomm* comm);

/**
 * \brief Creates a unique `comm_id` to be shared in a world of communicators.
 * \param ctx [gpucontext*] gpu context
 * \param comm_id [gpucommCliqueId*] pointer to instance containing id
 * \note Id is guaranteed to be unique across callers in a single host.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_gen_clique_id(gpucontext* ctx,
                                          gpucommCliqueId* comm_id);

/**
 * \brief Returns total number of device/communicators participating in `comm`'s
 * world.
 * \param comm [gpucomm*] gpu communicator
 * \param gpucount [int*] pointer to number of gpus in `comm`'s world
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_get_count(gpucomm* comm, int* gpucount);

/**
 * \brief Returns rank of `comm` inside its world as defined by user upon
 * creation.
 * \param comm [gpucomm*] gpu communicator
 * \param rank [int*] pointer to `comm`'s rank
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_get_rank(gpucomm* comm, int* rank);

/**
 * \brief Reduce collective operation for ranks in a communicator world [buffer
 * level].
 * \param src [gpudata*] data in device's buffer to be reduced
 * \param offsrc [size_t] memory offset after which data is saved in buffer
 * `src`
 * \param dest [gpudata*] data in device's buffer to collect result
 * \param offdest [size_t] memory offset after which data will be saved in
 * buffer
 * `dest`
 * \param count [size_t] number of elements to be reduced in each array
 * \param typecode [int] code for elements' data type, see \ref enum
 * GPUARRAY_TYPES
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param root [int] rank in `comm` which will collect result
 * \param comm [gpucomm*] gpu communicator
 * \note Non root ranks can call this, using a NULL `dest`. In this case,
 * `offdest`
 * will not be used.
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_reduce(gpudata* src, size_t offsrc, gpudata* dest,
                                   size_t offdest, size_t count, int typecode,
                                   int opcode, int root, gpucomm* comm);

/**
 * \brief AllReduce collective operation for ranks in a communicator world
 * [buffer
 * level].
 *
 * Reduces data pointed by `src` using op operation and leaves identical copies
 * of
 * result in data pointed by `dest` on each rank of `comm`.
 *
 * \param src [gpudata*] data in device's buffer to be reduced
 * \param offsrc [size_t] memory offset after which data is saved in buffer
 * `src`
 * \param dest [gpudata*] data in device's buffer to collect result
 * \param offdest [size_t] memory offset after which data will be saved in
 * buffer
 * `dest`
 * \param count [size_t] number of elements to be reduced in each array
 * \param typecode [int] code for elements' data type, see \ref enum
 * GPUARRAY_TYPES
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_all_reduce(gpudata* src, size_t offsrc,
                                       gpudata* dest, size_t offdest,
                                       size_t count, int typecode, int opcode,
                                       gpucomm* comm);

/**
 * \brief ReduceScatter collective operation for ranks in a communicator world
 * [buffer level].
 *
 * Reduces data pointed by `src` using `opcode` operation and leaves reduced
 * result
 * scattered over data pointed by `dest` in the user-defined rank order in
 * `comm`.
 *
 * \param src [gpudata*] data in device's buffer to be reduced
 * \param offsrc [size_t] memory offset after which data is saved in buffer
 * `src`
 * \param dest [gpudata*] data in device's buffer to collect scattered result
 * \param offdest [size_t] memory offset after which data will be saved in
 * buffer
 * `dest`
 * \param count [size_t] number of elements to be contained in result `dest`
 * \param typecode [int] code for elements' data type, see \ref enum
 * GPUARRAY_TYPES
 * \param opcode [int] reduce operation code, see \ref enum _gpucomm_reduce_ops
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_reduce_scatter(gpudata* src, size_t offsrc,
                                           gpudata* dest, size_t offdest,
                                           size_t count, int typecode,
                                           int opcode, gpucomm* comm);

/**
 * \brief Broadcast collective operation for ranks in a communicator world
 * [buffer
 * level].
 *
 * Copies data pointed by `array` to all ranks in `comm`.
 *
 * \param array [gpudata*] data in device's buffer to get copied or be received
 * \param offset [size_t] memory offset after which data in `array` begin
 * \param count [size_t] number of elements to be contained in `array`
 * \param typecode [int] code for elements' data type, see \ref enum
 * GPUARRAY_TYPES
 * \param root [int] rank in `comm` which broadcasts its array
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_broadcast(gpudata* array, size_t offset,
                                      size_t count, int typecode, int root,
                                      gpucomm* comm);

/**
 * \brief AllGather collective operation for ranks in a communicator world.
 *
 * Each rank receives all data pointed by `src` of every rank in the
 * user-defined
 * rank order in `comm`.
 *
 * \param src [gpudata*] data in device's buffer to be gathered
 * \param offsrc [size_t] memory offset after which data in `src` begin
 * \param dest [gpudata*] data in device's buffer to gather from all ranks
 * \param offdest [size_t] memory offset after which data in `dest` begin
 * \param count [size_t] number of elements to be gathered from each rank in
 * `src`
 * \param typecode [int] code for elements' data type, see \ref enum
 * GPUARRAY_TYPES
 * \param comm [gpucomm*] gpu communicator
 * \note Must be called separately for each rank in `comm`.
 * \return int error code, \ref GA_NO_ERROR if success
 */
GPUARRAY_PUBLIC int gpucomm_all_gather(gpudata* src, size_t offsrc,
                                       gpudata* dest, size_t offdest,
                                       size_t count, int typecode,
                                       gpucomm* comm);

#ifdef __cplusplus
}
#endif

#endif  // GPUARRAY_BUFFER_COLLECTIVES_H
