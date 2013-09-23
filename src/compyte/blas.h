#ifndef COMPYTE_BLAS_H
#define COMPYTE_BLAS_H

#incude <compyte/array.h>

COMPYTE_PUBLIC int GpuArray_sgemv(float alpha, GpuArray *A, GpuArray *X,
				  float beta, GpuArray *Y);

#endif
