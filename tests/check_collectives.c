#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include <check.h>
#include <mpi.h>

#include "gpuarray/array.h"
#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/collectives.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

#define ROOT_RANK 0
#define ND 2
#define ROWS 32
#define COLS 16

extern gpucontext* ctx;
extern gpucomm* comm;
extern int comm_ndev;
extern int comm_rank;

extern void setup_comm(void);
extern void teardown_comm(void);

#define STR(x) _STR(x)
#define _STR(x) #x
#define COUNT_ERRORS(A, B, M, N, res)           \
  do {                                          \
    int loci, locj;                             \
    res = 0;                                    \
    for (loci = 0; loci < (M); ++loci) {        \
      for (locj = 0; locj < (N); ++locj) {      \
        if ((A)[loci][locj] != (B)[loci][locj]) \
          res++;                                \
      }                                         \
    }                                           \
  } while (0)

/*******************************************************************************
*               Test array functions for collective operations                *
*******************************************************************************/

#define INIT_ARRAYS(inrows, incols, outrows, outcols)                        \
  int(*A)[(incols)];                                                         \
  int(*RES)[(outcols)];                                                      \
  int(*EXP)[(outcols)];                                                      \
  size_t indims[ND];                                                         \
  size_t outdims[ND];                                                        \
  const ssize_t outstrds[ND] = {sizeof(*RES), sizeof(int)};                  \
  int err;                                                                   \
  size_t i, j, outsize;                                                      \
  GpuArray Adev;                                                             \
  GpuArray RESdev;                                                           \
                                                                             \
  A = (int(*)[(incols)])calloc((inrows), sizeof(*A));                        \
  if (A == NULL)                                                             \
    ck_abort_msg("system memory allocation failed");                         \
  RES = (int(*)[(outcols)])calloc((outrows), sizeof(*RES));                  \
  if (RES == NULL)                                                           \
    ck_abort_msg("system memory allocation failed");                         \
  EXP = (int(*)[(outcols)])calloc((outrows), sizeof(*EXP));                  \
  if (EXP == NULL)                                                           \
    ck_abort_msg("system memory allocation failed");                         \
  indims[0] = (inrows);                                                      \
  indims[1] = (incols);                                                      \
  outdims[0] = (outrows);                                                    \
  outdims[1] = (outcols);                                                    \
  outsize = outdims[0] * outstrds[0];                                        \
                                                                             \
  for (i = 0; i < indims[0]; ++i)                                            \
    for (j = 0; j < indims[1]; ++j)                                          \
      A[i][j] = comm_rank + 2;                                               \
                                                                             \
  err = GpuArray_empty(&Adev, ctx, GA_INT, ND, indims, GA_C_ORDER);          \
  ck_assert_int_eq(err, GA_NO_ERROR);                                        \
  err = GpuArray_write(&Adev, A, sizeof(*A) * inrows);                       \
  ck_assert_int_eq(err, GA_NO_ERROR);                                        \
  err = GpuArray_empty(&RESdev, ctx, GA_INT, ND, outdims, GA_C_ORDER);       \
  ck_assert_int_eq(err, GA_NO_ERROR);

#define DESTROY_ARRAYS()   \
  GpuArray_clear(&RESdev); \
  GpuArray_clear(&Adev);   \
  free(A);                 \
  free(RES);               \
  free(EXP);

/**
 * \note Untested for `not proper element count` , `not agreeing typecode`, `not
 * aligned`.
 */
START_TEST(test_GpuArray_reduce) {
  int res;
  INIT_ARRAYS(ROWS, COLS, ROWS, COLS);

  if (comm_rank == ROOT_RANK) {
    err = GpuArray_reduce(&Adev, &RESdev, GA_SUM, ROOT_RANK, comm);
    ck_assert_int_eq(err, GA_NO_ERROR);
    GpuArray_sync(&RESdev);
    GpuArray_sync(&Adev);
  } else {
    err = GpuArray_reduce_from(&Adev, GA_SUM, ROOT_RANK, comm);
    ck_assert_int_eq(err, GA_NO_ERROR);
    GpuArray_sync(&Adev);
  }

  err = MPI_Reduce(A, EXP, ROWS * COLS, MPI_INT, MPI_SUM, ROOT_RANK,
                   MPI_COMM_WORLD);
  ck_assert_msg(err == MPI_SUCCESS, "openmpi error: cannot produced expected");

  if (comm_rank == ROOT_RANK) {
    err = GpuArray_read(RES, outsize, &RESdev);
    ck_assert_int_eq(err, GA_NO_ERROR);
    COUNT_ERRORS(RES, EXP, ROWS, COLS, res);
    ck_assert_msg(res == 0,
                  "GpuArray_reduce with %s op produced errors in %d places",
                  STR(GA_SUM), res);
  }

  DESTROY_ARRAYS();
}
END_TEST

/**
 * \note Untested for: `not proper element count` , `not agreeing typecode`,
 * `not
 * aligned`.
 */
START_TEST(test_GpuArray_all_reduce) {
  int res;
  INIT_ARRAYS(ROWS, COLS, ROWS, COLS);

  err = GpuArray_all_reduce(&Adev, &RESdev, GA_SUM, comm);
  ck_assert_int_eq(err, GA_NO_ERROR);
  GpuArray_sync(&RESdev);
  GpuArray_sync(&Adev);

  err = MPI_Allreduce(A, EXP, ROWS * COLS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ck_assert_msg(err == MPI_SUCCESS, "openmpi error: cannot produced expected");

  err = GpuArray_read(RES, outsize, &RESdev);
  ck_assert_int_eq(err, GA_NO_ERROR);
  COUNT_ERRORS(RES, EXP, ROWS, COLS, res);
  ck_assert_msg(res == 0,
                "GpuArray_all_reduce with %s op produced errors in %d places",
                STR(GA_SUM), res);

  DESTROY_ARRAYS();
}
END_TEST

/**
 * \note Untested for `not proper element count` , `not agreeing typecode`, `not
 * aligned`.
 */
START_TEST(test_GpuArray_reduce_scatter) {
  int res;
  int* recvcounts;
  // In order for C contiguous arrays to be combined/split successfully they
  // should
  // split along the smallest axis (the one with the bigger stride).
  INIT_ARRAYS(ROWS, COLS, ROWS / comm_ndev, COLS);

  err = GpuArray_reduce_scatter(&Adev, &RESdev, GA_SUM, comm);
  ck_assert_int_eq(err, GA_NO_ERROR);
  GpuArray_sync(&RESdev);
  GpuArray_sync(&Adev);

  recvcounts = (int*)malloc(comm_ndev * sizeof(int));
  if (recvcounts == NULL)
    ck_abort_msg("system memory allocation failed");
  for (i = 0; i < (size_t)comm_ndev; ++i)
    recvcounts[i] = ROWS * COLS / comm_ndev;
  err =
      MPI_Reduce_scatter(A, EXP, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  free(recvcounts);
  ck_assert_msg(err == MPI_SUCCESS, "openmpi error: cannot produced expected");

  err = GpuArray_read(RES, outsize, &RESdev);
  ck_assert_int_eq(err, GA_NO_ERROR);
  COUNT_ERRORS(RES, EXP, ROWS / comm_ndev, COLS, res);
  ck_assert_msg(
      res == 0,
      "GpuArray_reduce_scatter with %s op produced errors in %d places",
      STR(GA_SUM), res);

  DESTROY_ARRAYS();
}
END_TEST

/**
 * \note Untested for `not aligned`.
 */
START_TEST(test_GpuArray_broadcast) {
  int res;
  INIT_ARRAYS(ROWS, COLS, ROWS, COLS);

  for (i = 0; i < indims[0]; ++i)
    for (j = 0; j < indims[1]; ++j)
      EXP[i][j] = A[i][j];

  err = GpuArray_broadcast(&Adev, ROOT_RANK, comm);
  ck_assert_int_eq(err, GA_NO_ERROR);
  GpuArray_sync(&Adev);

  err = MPI_Bcast(EXP, ROWS * COLS, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
  ck_assert_msg(err == MPI_SUCCESS, "openmpi error: cannot produced expected");

  err = GpuArray_read(RES, outsize, &Adev);
  ck_assert_int_eq(err, GA_NO_ERROR);
  COUNT_ERRORS(RES, EXP, ROWS, COLS, res);
  ck_assert_msg(res == 0, "GpuArray_broadcast produced errors in %d places",
                res);

  DESTROY_ARRAYS();
}
END_TEST

/**
 * \note Untested for `not proper element count` , `not agreeing typecode`, `not
 * aligned`.
 */
START_TEST(test_GpuArray_all_gather) {
  int res;
  // In order for C contiguous arrays to be combined/split successfully they
  // should
  // split along the smallest axis (the one with the bigger stride).
  INIT_ARRAYS(ROWS / comm_ndev, COLS, ROWS, COLS);

  err = GpuArray_all_gather(&Adev, &RESdev, comm);
  ck_assert_int_eq(err, GA_NO_ERROR);
  GpuArray_sync(&RESdev);
  GpuArray_sync(&Adev);

  err = MPI_Allgather(A, ROWS * COLS / comm_ndev, MPI_INT, EXP,
                      ROWS * COLS / comm_ndev, MPI_INT, MPI_COMM_WORLD);
  ck_assert_msg(err == MPI_SUCCESS, "openmpi error: cannot produced expected");

  err = GpuArray_read(RES, outsize, &RESdev);
  ck_assert_int_eq(err, GA_NO_ERROR);
  COUNT_ERRORS(RES, EXP, ROWS, COLS, res);
  ck_assert_msg(res == 0, "GpuArray_all_gather produced errors in %d places",
                res);

  DESTROY_ARRAYS();
}
END_TEST

Suite* get_suite(void) {
  Suite* s = suite_create("collectives");
  TCase* tc = tcase_create("API");
  tcase_add_checked_fixture(tc, setup_comm, teardown_comm);
  tcase_add_test(tc, test_GpuArray_reduce);
  tcase_add_test(tc, test_GpuArray_all_reduce);
  tcase_add_test(tc, test_GpuArray_reduce_scatter);
  tcase_add_test(tc, test_GpuArray_broadcast);
  tcase_add_test(tc, test_GpuArray_all_gather);
  suite_add_tcase(s, tc);
  return s;
}
