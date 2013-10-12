#include "private.h"
#include "private_opencl.h"

#include <clBLAS.h>

#include "compyte/buffer_blas.h"
#include "compyte/error.h"

static inline clblasOrder convO(cb_order order) {
  switch (order) {
  case cb_row:
    return clblasRowMajor;
  case cb_column:
    return clblasColumnMajor;
  default:
    return -1;
  }
}

static inline clblasTranspose convT(cb_transpose trans) {
  switch (trans) {
  case cb_no_trans:
    return clblasNoTrans;
  case cb_trans:
    return clblasTrans;
  case cb_conj_trans:
    return clblasConjTrans;
  default:
    return -1;
  }
}

static unsigned int refcnt = 0;

static int setup(void *c) {
  cl_ctx *ctx = (cl_ctx *)c;
  clblasStatus err;

  if (refcnt == 0) {
    err = clblasSetup();
    if (err != clblasSuccess)
      return GA_BLAS_ERROR;
  }

  if (ctx->blas_handle == NULL)
    ctx->blas_handle = &refcnt;
  refcnt++;
  return GA_NO_ERROR;
}

static void teardown(void *c) {
  cl_ctx *ctx = (cl_ctx *)c;
  if (ctx->blas_handle != NULL) {
    ctx->blas_handle = NULL;
    refcnt--;
  }
  if (refcnt == 0)
    clblasTeardown();
}

#define NAME clblas

#define FETCH_CONTEXT(A) cl_ctx *ctx = (A)->ctx
#define FUNC_DECLS    \
  clblasStatus err;   \
  cl_uint num_ev = 0; \
  cl_event evl[3];    \
  cl_event ev

#define ARRAY_INIT(A)                           \
  if (A->ev != NULL)                            \
    evl[num_ev++] = A->ev

#define ARRAY_FINI(A)                           \
  if (A->ev != NULL)                            \
    clReleaseEvent(A->ev);                      \
  A->ev = ev;                                   \
  clRetainEvent(A->ev)

#define PRE_CALL err =
#define PREFIX(typec, TYPEC) clblas ## TYPEC
#define TRANS(tr) convT(tr)
#define ARRAY(A, dtype) A->buf, off ## A
#define TRAIL_ARGS , 1, &ctx->q, num_ev, num_ev == 0 ? NULL : evl, &ev

#define POST_CALL                               \
  if (err != clblasSuccess)                     \
    return GA_BLAS_ERROR

#define ORDER convO(order),

#include "generic_blas.inc.c"
