#include <stdlib.h>
#include "gpuarray/blas.h"
#include "gpuarray/buffer_blas.h"
#include "gpuarray/types.h"
#include "gpuarray/util.h"
#include "gpuarray/error.h"

int GpuArray_rgemv(cb_transpose transA, double alpha, GpuArray *A,
                   GpuArray *X, double beta, GpuArray *Y, int nocopy) {
  GpuArray *Ap = A;
  GpuArray copyA;
  GpuArray *Xp = X;
  GpuArray copyX;
  GpuArray *Yp = Y;
  void *ctx;
  size_t elsize;
  size_t m, n, lda;
  cb_order o;
  int err;

  if (A->typecode != GA_HALF &&
      A->typecode != GA_FLOAT &&
      A->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

  if (A->nd != 2 || X->nd != 1 || Y->nd != 1 ||
      A->typecode != A->typecode || X->typecode != A->typecode ||
      Y->typecode != A->typecode)
    return GA_VALUE_ERROR;

  if (!(A->flags & GA_ALIGNED) || !(X->flags & GA_ALIGNED) ||
      !(Y->flags & GA_ALIGNED))
    return GA_UNALIGNED_ERROR;

  if (transA == cb_no_trans) {
    m = A->dimensions[0];
    n = A->dimensions[1];
  } else {
    m = A->dimensions[1];
    n = A->dimensions[0];
  }

  if (Y->dimensions[0] != m || X->dimensions[0] != n)
    return GA_VALUE_ERROR;

  m = A->dimensions[0];
  n = A->dimensions[1];

  elsize = gpuarray_get_elsize(A->typecode);

  if (!GpuArray_ISONESEGMENT(A)) {
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
  if (Y->strides[0] < 0) {
    err = GA_VALUE_ERROR;
    goto cleanup;
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

  ctx = gpudata_context(Ap->data);
  err = gpublas_setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch (Ap->typecode) {
  case GA_HALF:
    err = gpublas_hgemv(o, transA, m, n, (float)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (float)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
    break;
  case GA_FLOAT:
    err = gpublas_sgemv(o, transA, m, n, (float)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (float)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
    break;
  case GA_DOUBLE:
    err = gpublas_dgemv(o, transA, m, n, (double)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (double)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
    break;
  }
 cleanup:
  if (Ap == &copyA)
    GpuArray_clear(&copyA);
  if (Xp == &copyX)
    GpuArray_clear(&copyX);
  return err;
}

int GpuArray_rgemm(cb_transpose transA, cb_transpose transB, double alpha,
                   GpuArray *A, GpuArray *B, double beta, GpuArray *C,
                   int nocopy) {
  GpuArray *Ap = A;
  GpuArray copyA;
  GpuArray *Bp = B;
  GpuArray copyB;
  GpuArray *Cp = C;
  void *ctx;
  size_t elsize;
  size_t m, n, k, lda, ldb, ldc;
  cb_order o;
  int err;

  if (A->typecode != GA_HALF && A->typecode != GA_FLOAT &&
      A->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

  if (A->nd != 2 || B->nd != 2 || C->nd != 2 ||
      A->typecode != A->typecode || B->typecode != A->typecode ||
      C->typecode != A->typecode)
    return GA_VALUE_ERROR;

  if (!(A->flags & GA_ALIGNED) || !(B->flags & GA_ALIGNED) ||
      !(C->flags & GA_ALIGNED))
    return GA_UNALIGNED_ERROR;

  if (transA == cb_no_trans) {
    m = A->dimensions[0];
    k = A->dimensions[1];
  } else {
    m = A->dimensions[1];
    k = A->dimensions[0];
  }

  if (transB == cb_no_trans) {
    n = B->dimensions[1];
    if (B->dimensions[0] != k)
      return GA_VALUE_ERROR;
  } else {
    n = B->dimensions[0];
    if (B->dimensions[1] != k)
      return GA_VALUE_ERROR;
  }

  if (C->dimensions[0] != m || C->dimensions[1] != n)
    return GA_VALUE_ERROR;

  elsize = gpuarray_get_elsize(A->typecode);

  if (!GpuArray_ISONESEGMENT(A)) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyA, A, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Ap = &copyA;
    }
  }
  if (!GpuArray_ISONESEGMENT(B)) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyB, B, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Bp = &copyB;
    }
  }
  if (!GpuArray_ISONESEGMENT(C)) {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  if (Cp->flags & GA_F_CONTIGUOUS) {
    o = cb_fortran;
    ldc = Cp->dimensions[0];
  } else if (Cp->flags & GA_C_CONTIGUOUS) {
    o = cb_c;
    ldc = Cp->dimensions[1];
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }
  if (Ap->flags & GA_F_CONTIGUOUS) {
    lda = Ap->dimensions[0];
    if (o == cb_c) {
      if (transA == cb_no_trans)
        transA = cb_trans;
      else
        transA = cb_no_trans;
    }
  } else if (Ap->flags & GA_C_CONTIGUOUS) {
    lda = Ap->dimensions[1];
    if (o == cb_fortran) {
      if (transA == cb_no_trans)
        transA = cb_trans;
      else
        transA = cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }
  if (Bp->flags & GA_F_CONTIGUOUS) {
    ldb = Bp->dimensions[0];
    if (o == cb_c) {
      if (transB == cb_no_trans)
        transB = cb_trans;
      else
        transB = cb_no_trans;
    }
  } else if (Bp->flags & GA_C_CONTIGUOUS) {
    ldb = Bp->dimensions[1];
    if (o == cb_fortran) {
      if (transB == cb_no_trans)
        transB = cb_trans;
      else
        transB = cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  ctx = gpudata_context(Ap->data);
  err = gpublas_setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch (Ap->typecode) {
  case GA_HALF:
      err = gpublas_hgemm(o, transA, transB, m, n, k, (float)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (float)beta, Cp->data, Cp->offset / elsize, ldc);
    break;
  case GA_FLOAT:
    err = gpublas_sgemm(o, transA, transB, m, n, k, (float)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (float)beta, Cp->data, Cp->offset / elsize, ldc);
    break;
  case GA_DOUBLE:
    err = gpublas_dgemm(o, transA, transB, m, n, k, (double)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (double)beta, Cp->data, Cp->offset / elsize, ldc);
    break;
  }

 cleanup:
  if (Ap == &copyA)
    GpuArray_clear(&copyA);
  if (Bp == &copyB)
    GpuArray_clear(&copyB);
  return err;
}

int GpuArray_rger(double alpha, GpuArray *X, GpuArray *Y, GpuArray *A,
                  int nocopy) {
  GpuArray *Xp = X;
  GpuArray copyX;
  GpuArray *Yp = Y;
  GpuArray copyY;
  GpuArray *Ap = A;
  void *ctx;
  size_t elsize;
  size_t m, n, lda;
  cb_order o;
  int err;

  if (X->typecode != GA_HALF && X->typecode != GA_FLOAT &&
      X->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

  if (X->nd != 1 || Y->nd != 1 || A->nd != 2 ||
      X->typecode != X->typecode || Y->typecode != X->typecode ||
      A->typecode != X->typecode)
    return GA_VALUE_ERROR;

  if (!(X->flags & GA_ALIGNED) || !(Y->flags & GA_ALIGNED) ||
      !(A->flags & GA_ALIGNED))
    return GA_UNALIGNED_ERROR;

  m = X->dimensions[0];
  n = Y->dimensions[0];
  if (A->dimensions[0] != m || A->dimensions[1] != n)
    return GA_VALUE_ERROR;

  elsize = gpuarray_get_elsize(X->typecode);

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
  if (Y->strides[0] < 0) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyY, Y, GA_ANY_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Yp = &copyY;
    }
  }
  if (!GpuArray_ISONESEGMENT(A)) {
    err = GA_VALUE_ERROR;
    goto cleanup;
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

  ctx = gpudata_context(Xp->data);
  err = gpublas_setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch(Xp->typecode) {
  case GA_HALF:
      err = gpublas_hger(o, m, n, (float)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  case GA_FLOAT:
    err = gpublas_sger(o, m, n, (float)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  case GA_DOUBLE:
    err = gpublas_dger(o, m, n, (double)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  }

 cleanup:
  if (Xp == &copyX)
    GpuArray_clear(&copyX);
  if (Yp == &copyY)
    GpuArray_clear(&copyY);
  return err;
}

int GpuArray_rgemmBatch_3d(cb_transpose transA, cb_transpose transB, double alpha,
                           GpuArray *A, GpuArray *B, double beta, GpuArray *C,
                           int nocopy) {
  GpuArray *Ap = A;
  GpuArray copyA;
  GpuArray *Bp = B;
  GpuArray copyB;
  GpuArray *Cp = C;
  void *ctx;
  size_t elsize;
  size_t batchCount, m, n, k, lda, ldb, ldc;
  cb_order o;
  int err;
  gpudata **A_datas = NULL, **B_datas = NULL, **C_datas = NULL;
  size_t *A_offsets = NULL, *B_offsets = NULL, *C_offsets = NULL;
  size_t i;

  if (A->typecode != GA_FLOAT && A->typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;

  if (A->nd != 3 || B->nd != 3 || C->nd != 3 ||
      A->typecode != A->typecode || B->typecode != A->typecode ||
      C->typecode != A->typecode)
    return GA_VALUE_ERROR;

  if (!(A->flags & GA_ALIGNED) || !(B->flags & GA_ALIGNED) ||
      !(C->flags & GA_ALIGNED))
    return GA_UNALIGNED_ERROR;

  batchCount = A->dimensions[0];
  if (B->dimensions[0] != batchCount || C->dimensions[0] != batchCount)
    return GA_VALUE_ERROR;

  if (transA == cb_no_trans) {
    m = A->dimensions[1];
    k = A->dimensions[2];
  } else {
    m = A->dimensions[2];
    k = A->dimensions[1];
  }

  if (transB == cb_no_trans) {
    n = B->dimensions[2];
    if (B->dimensions[1] != k)
      return GA_VALUE_ERROR;
  } else {
    n = B->dimensions[1];
    if (B->dimensions[2] != k)
      return GA_VALUE_ERROR;
  }

  if (C->dimensions[1] != m || C->dimensions[2] != n)
    return GA_VALUE_ERROR;

  elsize = gpuarray_get_elsize(A->typecode);

  // FIXME: these conditions are overly restrictive; the first axis need not be contiguous
  if (!GpuArray_ISONESEGMENT(A)) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyA, A, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Ap = &copyA;
    }
  }
  if (!GpuArray_ISONESEGMENT(B)) {
    if (nocopy)
      return GA_COPY_ERROR;
    else {
      err = GpuArray_copy(&copyB, B, GA_F_ORDER);
      if (err != GA_NO_ERROR)
	goto cleanup;
      Bp = &copyB;
    }
  }
  if (!GpuArray_ISONESEGMENT(C)) {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  if (Cp->flags & GA_F_CONTIGUOUS) {
    o = cb_fortran;
    ldc = Cp->dimensions[1];
  } else if (Cp->flags & GA_C_CONTIGUOUS) {
    o = cb_c;
    ldc = Cp->dimensions[2];
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }
  if (Ap->flags & GA_F_CONTIGUOUS) {
    lda = Ap->dimensions[1];
    if (o == cb_c) {
      if (transA == cb_no_trans)
        transA = cb_trans;
      else
        transA = cb_no_trans;
    }
  } else if (Ap->flags & GA_C_CONTIGUOUS) {
    lda = Ap->dimensions[2];
    if (o == cb_fortran) {
      if (transA == cb_no_trans)
        transA = cb_trans;
      else
        transA = cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }
  if (Bp->flags & GA_F_CONTIGUOUS) {
    ldb = Bp->dimensions[1];
    if (o == cb_c) {
      if (transB == cb_no_trans)
        transB = cb_trans;
      else
        transB = cb_no_trans;
    }
  } else if (Bp->flags & GA_C_CONTIGUOUS) {
    ldb = Bp->dimensions[2];
    if (o == cb_fortran) {
      if (transB == cb_no_trans)
        transB = cb_trans;
      else
        transB = cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  ctx = gpudata_context(Ap->data);
  err = gpublas_setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  A_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));
  B_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));
  C_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));

  A_offsets = (size_t*)malloc(batchCount * sizeof(size_t));
  B_offsets = (size_t*)malloc(batchCount * sizeof(size_t));
  C_offsets = (size_t*)malloc(batchCount * sizeof(size_t));

  for (i = 0; i < batchCount; i++) {
    A_datas[i] = Ap->data;
    B_datas[i] = Bp->data;
    C_datas[i] = Cp->data;
    A_offsets[i] = (Ap->offset + i * Ap->strides[0]) / elsize;
    B_offsets[i] = (Bp->offset + i * Bp->strides[0]) / elsize;
    C_offsets[i] = (Cp->offset + i * Cp->strides[0]) / elsize;
  }

  switch (C->typecode) {
  case GA_HALF:
    err = gpublas_hgemmBatch(o, transA, transB, m, n, k, (float)alpha,
                             A_datas, A_offsets, lda,
                             B_datas, B_offsets, ldb,
                             (float)beta,
                             C_datas, C_offsets, ldc, batchCount, 0);
    break;
  case GA_FLOAT:
    err = gpublas_sgemmBatch(o, transA, transB, m, n, k, (float)alpha,
                             A_datas, A_offsets, lda,
                             B_datas, B_offsets, ldb,
                             (float)beta,
                             C_datas, C_offsets, ldc, batchCount, 0);
    break;
  case GA_DOUBLE:
    err = gpublas_dgemmBatch(o, transA, transB, m, n, k, (double)alpha,
                             A_datas, A_offsets, lda,
                             B_datas, B_offsets, ldb,
                             (double)beta,
                             C_datas, C_offsets, ldc, batchCount, 0);
    break;
  }

  cleanup:
  free(A_datas); free(B_datas); free(C_datas);
  free(A_offsets); free(B_offsets); free(C_offsets);
  if (Ap == &copyA)
    GpuArray_clear(&copyA);
  if (Bp == &copyB)
    GpuArray_clear(&copyB);
  return err;
}
