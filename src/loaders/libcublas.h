#ifndef LOADER_LIBCUBLAS_H
#define LOADER_LIBCUBLAS_H

#include "util/error.h"
//TODO: how to have it work with align?
typedef struct {//__align__(2) {
  unsigned short x;
} __half;


/** @cond NEVER */

#ifdef _WIN32
#define CUBLASWINAPI __stdcall
#else
#define CUBLASWINAPI
#endif

typedef enum cudaDataType_t
{
  CUDA_R_16F= 2, // real as a half
  CUDA_C_16F= 6, // complex as a pair of half numbers
  CUDA_R_32F= 0, // real as a float
  CUDA_C_32F= 4, // complex as a pair of float numbers
  CUDA_R_64F= 1, // real as a double
  CUDA_C_64F= 5, // complex as a pair of double numbers
  CUDA_R_8I= 3,  // real as a signed char
  CUDA_C_8I= 7,   // complex as a pair of signed char numbers
  CUDA_R_8U= 8,  // real as a unsigned char
  CUDA_C_8U= 9,  // complex as a pair of unsigned char numbers
  CUDA_R_32I= 10,  // real as a signed int
  CUDA_C_32I= 11,  // complex as a pair of signed int numbers
  CUDA_R_32U= 12,  // real as a unsigned int
  CUDA_C_32U= 13   // complex as a pair of unsigned int numbers
} cudaDataType;

typedef cudaDataType cudaDataType_t;

typedef enum {
    CUBLAS_GEMM_DFALT               = -1,
    CUBLAS_GEMM_ALGO0               =  0,
    CUBLAS_GEMM_ALGO1               =  1,
    CUBLAS_GEMM_ALGO2               =  2,
    CUBLAS_GEMM_ALGO3               =  3,
    CUBLAS_GEMM_ALGO4               =  4,
    CUBLAS_GEMM_ALGO5               =  5,
    CUBLAS_GEMM_ALGO6               =  6,
    CUBLAS_GEMM_ALGO7               =  7,
    CUBLAS_GEMM_ALGO8               =  8,
    CUBLAS_GEMM_ALGO9               =  9,
    CUBLAS_GEMM_ALGO10              =  10,
    CUBLAS_GEMM_ALGO11              =  11,
    CUBLAS_GEMM_ALGO12              =  12,
    CUBLAS_GEMM_ALGO13              =  13,
    CUBLAS_GEMM_ALGO14              =  14,
    CUBLAS_GEMM_ALGO15              =  15,
    CUBLAS_GEMM_ALGO16              =  16,
    CUBLAS_GEMM_ALGO17              =  17,
    CUBLAS_GEMM_DFALT_TENSOR_OP     =  99,
    CUBLAS_GEMM_ALGO0_TENSOR_OP     =  100,
    CUBLAS_GEMM_ALGO1_TENSOR_OP     =  101,
    CUBLAS_GEMM_ALGO2_TENSOR_OP     =  102
} cublasGemmAlgo_t;

typedef struct CUstream_st *cudaStream_t;

typedef enum {
  CUBLAS_STATUS_SUCCESS         =0,
  CUBLAS_STATUS_NOT_INITIALIZED =1,
  CUBLAS_STATUS_ALLOC_FAILED    =3,
  CUBLAS_STATUS_INVALID_VALUE   =7,
  CUBLAS_STATUS_ARCH_MISMATCH   =8,
  CUBLAS_STATUS_MAPPING_ERROR   =11,
  CUBLAS_STATUS_EXECUTION_FAILED=13,
  CUBLAS_STATUS_INTERNAL_ERROR  =14,
  CUBLAS_STATUS_NOT_SUPPORTED   =15,
  CUBLAS_STATUS_LICENSE_ERROR   =16
} cublasStatus_t;

typedef enum {
  CUBLAS_OP_N=0,
  CUBLAS_OP_T=1,
  CUBLAS_OP_C=2
} cublasOperation_t;

typedef enum {
  CUBLAS_POINTER_MODE_HOST   = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
  CUBLAS_ATOMICS_NOT_ALLOWED   = 0,
  CUBLAS_ATOMICS_ALLOWED       = 1
} cublasAtomicsMode_t;

typedef struct cublasContext *cublasHandle_t;

/** @endcond */

int load_libcublas(int major, int minor, error *e);

/** @cond NEVER */

#define DEF_PROC(name, args) typedef cublasStatus_t CUBLASWINAPI t##name args
#define DEF_PROC_V2(name, args) DEF_PROC(name, args)
#define DEF_PROC_OPT(name, args) DEF_PROC(name, args)

#include "libcublas.fn"

#undef DEF_PROC_OPT
#undef DEF_PROC_V2
#undef DEF_PROC

#define DEF_PROC(name, args) extern t##name *name
#define DEF_PROC_V2(name, args) DEF_PROC(name, args)
#define DEF_PROC_OPT(name, args) DEF_PROC(name, args)

#include "libcublas.fn"

#undef DEF_PROC_OPT
#undef DEF_PROC_V2
#undef DEF_PROC

/** @endcond */

#endif
