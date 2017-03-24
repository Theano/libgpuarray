#ifndef LOADER_LIBCLBLAS_H
#define LOADER_LIBCLBLAS_H

#include "util/error.h"
#include "libopencl.h"

typedef enum clblasOrder_ {
  clblasRowMajor,
  clblasColumnMajor
} clblasOrder;

typedef enum clblasTranspose_ {
  clblasNoTrans,
  clblasTrans,
  clblasConjTrans
} clblasTranspose;

typedef enum clblasStatus_ {
  clblasSuccess = CL_SUCCESS,
  /* Rest is not exposed from here */
  clblasNotImplemented = -1024,
  clblasNotInitialized,
  clblasInvalidMatA,
  clblasInvalidMatB,
  clblasInvalidMatC,
  clblasInvalidVecX,
  clblasInvalidVecY,
  clblasInvalidDim,
  clblasInvalidLeadDimA,
  clblasInvalidLeadDimB,
  clblasInvalidLeadDimC,
  clblasInvalidIncX,
  clblasInvalidIncY,
  clblasInsufficientMemMatA,
  clblasInsufficientMemMatB,
  clblasInsufficientMemMatC,
  clblasInsufficientMemVecX,
  clblasInsufficientMemVecY,
} clblasStatus;

int load_libclblas(error *);

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libclblas.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libclblas.fn"

#undef DEF_PROC

#endif
