#ifndef COMPYTE_BLAS_H
#define COMPYTE_BLAS_H

#include <compyte/buffer_blas.h>
#include <compyte/array.h>

COMPYTE_PUBLIC int GpuArray_sgemv(cb_transpose transA, float alpha,
				  GpuArray *A, GpuArray *X, float beta,
				  GpuArray *Y);

#endif
