#include "compyte/blas.h"
#include "compyte/buffer_blas.h"
#include "compyte/types.h"
#include "compyte/util.h"
#include "compyte/error.h"

int GpuArray_rgemv(cb_transpose t, double alpha, const GpuArray *A,
		   const GpuArray *X, double beta, GpuArray *Y, int nocopy) {
  const GpuArray *Ap = A, *Xp = X;
  compyte_blas_ops *blas;
  void *ctx;
  size_t elsize;
  size_t lda;
  GpuArray copyA, copyX;
  cb_order o;
  int err;

  if (A->typecode != GA_FLOAT && A->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

  if (A->nd != 2 || X->nd != 1 || Y->nd != 1 || X->typecode != A->typecode ||
      Y->typecode != A->typecode)
    return GA_VALUE_ERROR;

  if (!(A->flags & GA_ALIGNED) || !(X->flags & GA_ALIGNED) ||
      !(Y->flags & GA_ALIGNED))
    return GA_UNALIGNED_ERROR;

  if (t == cb_no_trans) {
    if (Ap->dimensions[1] != Xp->dimensions[0] ||
	Ap->dimensions[0] != Y->dimensions[0])
      return GA_VALUE_ERROR;
  } else {
    if (Ap->dimensions[1] != Y->dimensions[0] ||
	Ap->dimensions[0] != Xp->dimensions[0])
      return GA_VALUE_ERROR;
  }

  /* We never copy Y */
  if (Y->strides[0] < 0)
    return GA_VALUE_ERROR;

  elsize = compyte_get_elsize(A->typecode);

  if (!GpuArray_ISONESEGMENT(A) ||
      (A->dimensions[0] > 1 && A->strides[0] != elsize &&
       A->dimensions[1] > 1 && A->strides[1] != elsize) ||
      A->strides[0] < 0 || A->strides[1] < 0) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyA, A, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Ap = &copyA;
    }
  }

  if (X->strides[0] < 0) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyX, X, GA_ANY_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Xp = &copyX;
    }
  }

  if (Ap->flags & GA_F_CONTIGUOUS) {
    o = cb_fortran;
    lda = Ap->dimensions[0];
  } else if (Ap->flags & GA_C_CONTIGUOUS) {
    o = cb_c;
    lda = Ap->dimensions[1];
  } else {
    /* Might be worth looking at making degenerate matrices (1xn) work here. */
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  err = Ap->ops->property(NULL, Ap->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;
  err = Ap->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    goto cleanup;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  if (Ap->typecode == GA_FLOAT)
    err = blas->sgemv(o, t, Ap->dimensions[0], Ap->dimensions[1], (float)alpha,
		      Ap->data, Ap->offset / elsize, lda,
		      Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize,
		      (float)beta,
		      Y->data, Y->offset / elsize, Y->strides[0] / elsize);
  else
    err = blas->dgemv(o, t, Ap->dimensions[0], Ap->dimensions[1], alpha,
		      Ap->data, Ap->offset / elsize, lda,
		      Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize,
		      beta,
		      Y->data, Y->offset / elsize, Y->strides[0] / elsize);
 cleanup:
  if (Ap == &copyA)
    GpuArray_clear(&copyA);
  if (Xp == &copyX)
    GpuArray_clear(&copyX);
  return err;
}
