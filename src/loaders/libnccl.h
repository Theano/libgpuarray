#ifndef LOADER_LIBNCCL_H
#define LOADER_LIBNCCL_H

#include "util/error.h"

typedef struct CUstream_st *cudaStream_t;
typedef struct ncclComm* ncclComm_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

typedef enum { ncclSuccess = 0 } ncclResult_t;

typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               nccl_NUM_OPS   = 4 } ncclRedOp_t;

/* Data types */
typedef enum { ncclChar       = 0,
               ncclInt        = 1,
               ncclHalf       = 2,
               ncclFloat      = 3,
               ncclDouble     = 4,
               ncclInt64      = 5,
               ncclUint64     = 6,
               nccl_NUM_TYPES = 7 } ncclDataType_t;

int load_libnccl(error *e);

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libnccl.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libnccl.fn"

#undef DEF_PROC

#endif
