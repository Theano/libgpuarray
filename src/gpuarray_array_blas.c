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
  gpuarray_blas_ops *blas;
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

  err = Ap->ops->property(NULL, Ap->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;
  err = Ap->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    goto cleanup;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch (Ap->typecode) {
  case GA_HALF:
    if (blas->hgemv == NULL)
      err = GA_DEVSUP_ERROR;
    else
      err = blas->hgemv(o, transA, m, n, (float)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (float)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
    break;
  case GA_FLOAT:
    err = blas->sgemv(o, transA, m, n, (float)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (float)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
    break;
  case GA_DOUBLE:
    err = blas->dgemv(o, transA, m, n, (double)alpha, Ap->data, Ap->offset / elsize, lda, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, (double)beta, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize);
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
  gpuarray_blas_ops *blas;
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

  err = Ap->ops->property(NULL, Ap->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;
  err = Ap->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    goto cleanup;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch (Ap->typecode) {
  case GA_HALF:
    if (blas->hgemm == NULL)
      err = GA_DEVSUP_ERROR;
    else
      err = blas->hgemm(o, transA, transB, m, n, k, (float)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (float)beta, Cp->data, Cp->offset / elsize, ldc);
    break;
  case GA_FLOAT:
    err = blas->sgemm(o, transA, transB, m, n, k, (float)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (float)beta, Cp->data, Cp->offset / elsize, ldc);
    break;
  case GA_DOUBLE:
    err = blas->dgemm(o, transA, transB, m, n, k, (double)alpha, Ap->data, Ap->offset / elsize, lda, Bp->data, Bp->offset / elsize, ldb, (double)beta, Cp->data, Cp->offset / elsize, ldc);
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
  gpuarray_blas_ops *blas;
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

  err = Xp->ops->property(NULL, Xp->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;
  err = Xp->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
  if (err != GA_NO_ERROR)
    goto cleanup;

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch(Xp->typecode) {
  case GA_HALF:
    if (blas->hger == NULL)
      err = GA_DEVSUP_ERROR;
    else
      err = blas->hger(o, m, n, (float)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  case GA_FLOAT:
    err = blas->sger(o, m, n, (float)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  case GA_DOUBLE:
    err = blas->dger(o, m, n, (double)alpha, Xp->data, Xp->offset / elsize, Xp->strides[0] / elsize, Yp->data, Yp->offset / elsize, Yp->strides[0] / elsize, Ap->data, Ap->offset / elsize, lda);
    break;
  }

 cleanup:
  if (Xp == &copyX)
    GpuArray_clear(&copyX);
  if (Yp == &copyY)
    GpuArray_clear(&copyY);
  return err;
}

int GpuArray_rgemmBatch(cb_transpose transA, cb_transpose transB, double alpha,
                        GpuArray **A, GpuArray **B, double beta, GpuArray **C,
                        size_t batchCount, int nocopy) {
  GpuArray **A_copies = 0, **B_copies = 0, **C_copies = 0;
  gpudata **A_datas = 0, **B_datas = 0, **C_datas = 0;
  size_t *A_offsets = 0, *B_offsets = 0, *C_offsets = 0;
  gpuarray_blas_ops *blas;
  void *ctx;
  size_t elsize;
  int typecode;
  size_t m, n, k, lda, ldb, ldc;
  cb_order o;
  int err;

  if (batchCount == 0)
    return GA_NO_ERROR;

  typecode = A[0]->typecode;
  if (typecode != GA_HALF && typecode != GA_FLOAT && typecode != GA_DOUBLE)
    return GA_INVALID_ERROR;
  elsize = gpuarray_get_elsize(typecode);

  /* determine shape */
  if (transA == cb_no_trans) {
    m = A[0]->dimensions[0];
    k = A[0]->dimensions[1];
  } else {
    m = A[0]->dimensions[1];
    k = A[0]->dimensions[0];
  }

  if (transB == cb_no_trans) {
    n = B[0]->dimensions[1];
  } else {
    n = B[0]->dimensions[0];
  }

  /* determine order */
  if (C[0]->flags & GA_F_CONTIGUOUS) {
    o = cb_fortran;
    ldc = C[0]->dimensions[0];
  } else if (C[0]->flags & GA_C_CONTIGUOUS) {
    o = cb_c;
    ldc = C[0]->dimensions[1];
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  if (A[0]->flags & GA_F_CONTIGUOUS) {
    lda = A[0]->dimensions[0];
    if (o == cb_c) {
      transA = transA == cb_no_trans ? cb_trans : cb_no_trans;
    }
  } else if (A[0]->flags & GA_C_CONTIGUOUS) {
    lda = A[0]->dimensions[1];
    if (o == cb_fortran) {
      transA = transA == cb_no_trans ? cb_trans : cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  if (B[0]->flags & GA_F_CONTIGUOUS) {
    ldb = B[0]->dimensions[0];
    if (o == cb_c) {
      transB = transB == cb_no_trans ? cb_trans : cb_no_trans;
    }
  } else if (B[0]->flags & GA_C_CONTIGUOUS) {
    ldb = B[0]->dimensions[1];
    if (o == cb_fortran) {
      transB = transB == cb_no_trans ? cb_trans : cb_no_trans;
    }
  } else {
    err = GA_VALUE_ERROR;
    goto cleanup;
  }

  /* some elements of A, B and C may need to be reallocated
     contiguously; take care of the copies so that we may free them
     later. */
  A_copies = (GpuArray**)calloc(batchCount, sizeof(GpuArray*));
  B_copies = (GpuArray**)calloc(batchCount, sizeof(GpuArray*));
  if (!(A_copies && B_copies)) {
    /* TODO: don't care if nocopy */
    err = GA_COPY_ERROR;
    goto cleanup;
  }

  A_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));
  B_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));
  C_datas = (gpudata**)malloc(batchCount * sizeof(gpudata*));
  A_offsets = (size_t*)malloc(batchCount * sizeof(size_t));
  B_offsets = (size_t*)malloc(batchCount * sizeof(size_t));
  C_offsets = (size_t*)malloc(batchCount * sizeof(size_t));

  for (int i = 0; i < batchCount; i++) {
    if (A[i]->nd != 2 || B[i]->nd != 2 || C[i]->nd != 2 ||
        A[i]->typecode != typecode || B[i]->typecode != typecode ||
        C[i]->typecode != typecode) {
      err = GA_VALUE_ERROR;
      goto cleanup;
    }

    if (!(A[i]->flags & GA_ALIGNED) || !(B[i]->flags & GA_ALIGNED) ||
        !(C[i]->flags & GA_ALIGNED)) {
      err = GA_UNALIGNED_ERROR;
      goto cleanup;
    }

    /* ensure shapes match */
    if (transA == cb_no_trans) {
      if (A[i]->dimensions[0] != m || A[i]->dimensions[1] != k) {
        err = GA_VALUE_ERROR;
        goto cleanup;
      }
    } else {
      if (A[i]->dimensions[0] != k || A[i]->dimensions[1] != m) {
        err = GA_VALUE_ERROR;
        goto cleanup;
      }
    }

    if (transB == cb_no_trans) {
      if (B[i]->dimensions[0] != k || B[i]->dimensions[1] != n) {
        err = GA_VALUE_ERROR;
        goto cleanup;
    } else {
      if (B[i]->dimensions[0] != n || B[i]->dimensions[1] != k) {
        err = GA_VALUE_ERROR;
        goto cleanup;
      }
    }

    if (C[i]->dimensions[0] != m || C[i]->dimensions[1] != n) {
      err = GA_VALUE_ERROR;
      goto cleanup;
    }

    /* reallocate contiguously if necessary */
    if (!GpuArray_ISONESEGMENT(A[i])) {
      if (nocopy) {
        err = GA_COPY_ERROR;
        goto cleanup;
      } else {
        A_copies[i] = (GpuArray*)malloc(sizeof(GpuArray));
        if (!A_copies[i])
          err = GA_COPY_ERROR;
          goto cleanup;
        err = GpuArray_copy(&A_copies[i], A[i], GA_F_ORDER);
        if (err != GA_NO_ERROR)
          goto cleanup;
        A[i] = A_copies[i];
      }
    }

    if (!GpuArray_ISONESEGMENT(B[i])) {
      if (nocopy) {
        err = GA_COPY_ERROR;
        goto cleanup;
      } else {
        B_copies[i] = (GpuArray*)malloc(sizeof(GpuArray));
        if (!B_copies[i])
          err = GA_COPY_ERROR;
          goto cleanup;
        err = GpuArray_copy(&B_copies[i], B[i], GA_F_ORDER);
        if (err != GA_NO_ERROR)
          goto cleanup;
        B[i] = B_copies[i];
      }
    }

    if (!GpuArray_ISONESEGMENT(C[i])) {
      err = GA_VALUE_ERROR;
      goto cleanup;
    }

    /* ensure orders match */
    if (A[i]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS) !=
        A[0]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS)) {
      err = GA_VALUE_ERROR;
      goto cleanup;
    if (B[i]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS) !=
        B[0]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS)) {
      err = GA_VALUE_ERROR;
      goto cleanup;
    if (C[i]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS) !=
        C[0]->flags & (GA_F_CONTIGUOUS | GA_C_CONTIGUOUS)) {
      err = GA_VALUE_ERROR;
      goto cleanup;

    err = A[i]->ops->property(NULL, A[i]->data, NULL, GA_BUFFER_PROP_CTX, &ctx);
    if (err != GA_NO_ERROR)
      goto cleanup;
    err = A[i]->ops->property(ctx, NULL, NULL, GA_CTX_PROP_BLAS_OPS, &blas);
    if (err != GA_NO_ERROR)
      goto cleanup;

    A_datas[i] = A[i]->data; A_offsets[i] = A[i]->offset / elsize;
    B_datas[i] = B[i]->data; B_offsets[i] = B[i]->offset / elsize;
    C_datas[i] = C[i]->data; C_offsets[i] = C[i]->offset / elsize;
  }

  err = blas->setup(ctx);
  if (err != GA_NO_ERROR)
    goto cleanup;

  switch (typecode) {
  case GA_HALF:
    if (blas->hgemm == NULL)
      err = GA_DEVSUP_ERROR;
    else
      err = blas->hgemmBatch(o, transA, transB, m, n, k, (float)alpha,
                             A_datas, A_offsets / elsize, lda,
                             B_datas, B_offsets / elsize, ldb,
                             (float)beta,
                             C_datas, C_offsets / elsize, ldc);
    break;
  case GA_FLOAT:
    err = blas->sgemmBatch(o, transA, transB, m, n, k, (float)alpha,
                           A_datas, A_offsets / elsize, lda,
                           B_datas, B_offsets / elsize, ldb,
                           (float)beta,
                           C_datas, C_offsets / elsize, ldc);
    break;
  case GA_DOUBLE:
    err = blas->dgemmBatch(o, transA, transB, m, n, k, (double)alpha,
                           A_datas, A_offsets / elsize, lda,
                           B_datas, B_offsets / elsize, ldb,
                           (double)beta,
                           C_datas, C_offsets / elsize, ldc);
    break;
  }

  cleanup:
  free(A_datas); free(B_datas); free(C_datas);
  free(A_offsets); free(B_offsets); free(C_offsets);
  if (A_copies) {
    for (int i = 0; i < batchCount; i++) {
      if (A_copies[i]) {
        GpuArray_clear(A_copies[i]);
        free(A_copies[i]);
        A_copies[i] = 0;
      }
    }
    free(A_copies);
  }
  if (B_copies) {
    for (int i = 0; i < batchCount; i++) {
      if (B_copies[i]) {
        GpuArray_clear(B_copies[i]);
        free(B_copies[i]);
        B_copies[i] = 0;
      }
    }
    free(B_copies);
  }
  return err;
}
