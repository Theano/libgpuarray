#include "compyte/blas.h"
#include "compyte/buffer_blas.h"
#include "compyte/types.h"
#include "compyte/util.h"
#include "compyte/error.h"

int GpuArray_sgemv(cb_transpose t, float alpha, GpuArray *A, GpuArray *X,
		   float beta, GpuArray *Y) {
  compyte_blas_ops *blas;
  void *ctx;
  size_t elsize;
  size_t n, m, lda;
  cb_order o;
  int err;

  if (A->nd != 2 || X->nd != 1 || Y->nd != 1 ||
      A->typecode != GA_FLOAT || X->typecode != GA_FLOAT ||
      Y->typecode != GA_FLOAT)
    return GA_VALUE_ERROR;

  if (A->flags & GA_F_CONTIGUOUS)
    o = cb_fortran;
  else if (A->flags & GA_C_CONTIGUOUS)
    o = cb_c;
  else
    return GA_VALUE_ERROR;

  if (t == cb_no_trans) {
    if (A->dimensions[1] != X->dimensions[0])
      return GA_VALUE_ERROR;
    m = A->dimensions[0];
    n = A->dimensions[1];
  } else {
    if (A->dimensions[1] != Y->dimensions[0])
      return GA_VALUE_ERROR;
    m = A->dimensions[1];
    n = A->dimensions[0];
  }

  if (o == cb_c) {
    lda = A->dimensions[1];
  } else {
    lda = A->dimensions[0];
  }

  elsize = compyte_get_elsize(GA_FLOAT);

  if (A->offset % elsize != 0 || X->offset % elsize != 0 ||
      Y->offset % elsize != 0 || X->strides[0] % elsize != 0 ||
      Y->strides[0] % elsize != 0)
    return GA_VALUE_ERROR;

  err = A->ops->property(NULL, A->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    return err;
  err = A->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    return err;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    return err;

  return blas->sgemv(o, t, m, n, alpha, A->data, A->offset / elsize,
		     lda, X->data, X->offset / elsize,
		     X->strides[0] / elsize, beta, Y->data, Y->offset / elsize,
		     Y->strides[0] / elsize);
}
