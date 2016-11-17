#ifndef LOADER_LIBCLBLAS_H
#define LOADER_LIBCLBLAS_H

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
} clblasStatus;

int load_libclblas(void);

#define DEF_PROC(ret, name, args) typedef ret t##name args

#include "libclblas.fn"

#undef DEF_PROC

#define DEF_PROC(ret, name, args) extern t##name *name

#include "libclblas.fn"

#undef DEF_PROC

#endif
