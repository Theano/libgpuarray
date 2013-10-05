#ifndef COMPYTE_BLAS_H
#define COMPYTE_BLAS_H

#include <compyte/buffer_blas.h>
#include <compyte/array.h>


COMPYTE_PUBLIC int GpuArray_rgemv(cb_transpose t, double alpha,
				  const GpuArray *A, const GpuArray *X,
				  double beta, GpuArray *Y, int nocopy);
#define GpuArray_sgemv GpuArray_rgemv
#define GpuArray_dgemv GpuArray_rgemv

#endif
