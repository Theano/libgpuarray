#ifndef LOADER_LIBNCCL_H
#define LOADER_LIBNCCL_H

#include "util/error.h"

/** @cond NEVER */

typedef struct CUstream_st *cudaStream_t;
typedef struct ncclComm* ncclComm_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

typedef enum { ncclSuccess = 0 } ncclResult_t;

/* Reduction operation selector */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclNumOps     = 4 } ncclRedOp_t;
/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclNumTypes   = 9 } ncclDataType_t;

/** @endcond */

int load_libnccl(error *e);

/* @cond NEVER */

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libnccl.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libnccl.fn"

#undef DEF_PROC

/** @endcond */

#endif
