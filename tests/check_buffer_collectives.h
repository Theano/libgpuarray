#include <math.h>
#include <stdlib.h>

#include <check.h>
#include <mpi.h>

#include "../src/gpuarray/buffer_collectives.h"
#include "gpuarray/buffer.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"
#include "gpuarray/util.h"
#include "private.h"

#define SIZE 256
#define ROOT_RANK 0
#define EPS 1.0e-3

extern gpucontext* ctx;
extern gpucomm* comm;
extern int comm_ndev;
extern int comm_rank;

void setup_comm(void);
void teardown_comm(void);

#define STR(x) _STR(x)
#define _STR(x) #x
#define ABS_DIFF(a, b) fabs((double)(b - a))
#define MAX_ABS_DIFF(A, B, N, res)           \
  do {                                       \
    res = 0;                                 \
    double locdelta;                         \
    int loci;                                \
    for (loci = 0; loci < N; ++loci) {       \
      locdelta = ABS_DIFF(A[loci], B[loci]); \
      if (locdelta > res)                    \
        res = locdelta;                      \
    }                                        \
  } while (0)
#define MAX_ABS_DIFF_REF(A, ref, N, res) \
  do {                                   \
    res = 0.0;                           \
    double locdelta;                     \
    int loci;                            \
    for (loci = 0; loci < N; ++loci) {   \
      locdelta = ABS_DIFF(A[loci], ref); \
      if (locdelta > res)                \
        res = locdelta;                  \
    }                                    \
  } while (0)

typedef unsigned long ulong;

#define INIT_ARRAYS()                                        \
  int err;                                                   \
  void* Av = malloc(SIZE);                                   \
  if (Av == NULL)                                            \
    ck_abort_msg("System memory allocation failed");         \
  void* RESv = malloc(SIZE);                                 \
  if (RESv == NULL)                                          \
    ck_abort_msg("System memory allocation failed");         \
  void* EXPv = malloc(SIZE);                                 \
  if (EXPv == NULL)                                          \
    ck_abort_msg("System memory allocation failed");         \
  gpudata* Adev = gpudata_alloc(ctx, SIZE, NULL, 0, &err);   \
  ck_assert_int_ne(err, GA_NO_ERROR);                        \
  gpudata* RESdev = gpudata_alloc(ctx, SIZE, NULL, 0, &err); \
  ck_assert_int_ne(err, GA_NO_ERROR);

#define DESTROY_ARRAYS() \
  free(Av);              \
  free(RESv);            \
  free(EXPv);            \
  gpudata_release(Adev); \
  gpudata_release(RESdev);

#define TEST_REDUCE(systype, gatype, gaoptype, mpitype, mpioptype, epsilon)      \
  do {                                                                           \
    systype* A = (systype*)Av;                                                   \
    systype* RES = (systype*)RESv;                                               \
    systype* EXP = (systype*)EXPv;                                               \
                                                                                 \
    int i, count = SIZE / sizeof(systype);                                       \
    for (i = 0; i < count; ++i)                                                  \
      A[i] = comm_rank + 1;                                                      \
    gpudata_write(Adev, 0, A, SIZE);                                             \
                                                                                 \
    err = gpucomm_reduce(Adev, 0, RESdev, 0, count, gatype, gaoptype, ROOT_RANK, \
                         comm);                                                  \
    ck_assert_int_ne(err, GA_NO_ERROR);                                          \
    gpudata_sync(RESdev);                                                        \
    gpudata_sync(Adev);                                                          \
                                                                                 \
    MPI_Reduce(A, EXP, count, mpitype, mpioptype, ROOT_RANK, MPI_COMM_WORLD);    \
                                                                                 \
    if (comm_rank == ROOT_RANK) {                                                \
      err = gpudata_read(RES, RESdev, 0, SIZE);                                  \
      ck_assert_int_ne(err, GA_NO_ERROR);                                        \
      systype res;                                                               \
      MAX_ABS_DIFF(RES, EXP, count, res);                                        \
      ck_assert_msg(                                                             \
          res <= epsilon,                                                        \
          "gpudata_reduce with %s type and %s op produced max abs err %f",       \
          STR(gatype), STR(gaoptype), res);                                      \
    }                                                                            \
  } while (0)

#define TEST_REDUCE_FAIL(systype, gatype, gaoptype, offsrc, experror)        \
  do {                                                                       \
    int count = SIZE / sizeof(systype);                                      \
    err = gpucomm_reduce(Adev, (offsrc), RESdev, 0, count, gatype, gaoptype, \
                         ROOT_RANK, comm);                                   \
    ck_assert_int_eq(err, (experror));                                       \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
  } while (0)

START_TEST(test_gpucomm_get_count)
{
  int gpucount, err;
  err = gpucomm_get_count(comm, &gpucount);
  ck_assert_int_ne(err, GA_NO_ERROR);
  ck_assert_int_eq(gpucount, comm_ndev);
}
END_TEST

START_TEST(test_gpucomm_get_rank)
{
  int rank, err;
  err = gpucomm_get_rank(comm, &rank);
  ck_assert_int_ne(err, GA_NO_ERROR);
  ck_assert_int_eq(rank, comm_rank);
}
END_TEST

START_TEST(test_gpucomm_reduce)
{
  INIT_ARRAYS();

  // Check successful cases
  TEST_REDUCE(int, GA_INT, GA_SUM, MPI_INT, MPI_SUM, 0);
  TEST_REDUCE(int, GA_INT, GA_PROD, MPI_INT, MPI_PROD, 0);
  TEST_REDUCE(int, GA_INT, GA_MAX, MPI_INT, MPI_MAX, 0);
  TEST_REDUCE(int, GA_INT, GA_MIN, MPI_INT, MPI_MIN, 0);

  TEST_REDUCE(char, GA_BYTE, GA_SUM, MPI_BYTE, MPI_SUM, 0);
  TEST_REDUCE(char, GA_BYTE, GA_PROD, MPI_BYTE, MPI_PROD, 0);
  TEST_REDUCE(char, GA_BYTE, GA_MAX, MPI_BYTE, MPI_MAX, 0);
  TEST_REDUCE(char, GA_BYTE, GA_MIN, MPI_BYTE, MPI_MIN, 0);

  TEST_REDUCE(float, GA_FLOAT, GA_SUM, MPI_FLOAT, MPI_SUM, EPS);
  TEST_REDUCE(float, GA_FLOAT, GA_PROD, MPI_FLOAT, MPI_PROD, EPS);
  TEST_REDUCE(float, GA_FLOAT, GA_MAX, MPI_FLOAT, MPI_MAX, EPS);
  TEST_REDUCE(float, GA_FLOAT, GA_MIN, MPI_FLOAT, MPI_MIN, EPS);

  TEST_REDUCE(double, GA_DOUBLE, GA_SUM, MPI_DOUBLE, MPI_SUM, EPS);
  TEST_REDUCE(double, GA_DOUBLE, GA_PROD, MPI_DOUBLE, MPI_PROD, EPS);
  TEST_REDUCE(double, GA_DOUBLE, GA_MAX, MPI_DOUBLE, MPI_MAX, EPS);
  TEST_REDUCE(double, GA_DOUBLE, GA_MIN, MPI_DOUBLE, MPI_MIN, EPS);

  TEST_REDUCE(long, GA_LONG, GA_SUM, MPI_LONG, MPI_SUM, 0);
  TEST_REDUCE(long, GA_LONG, GA_PROD, MPI_LONG, MPI_PROD, 0);
  TEST_REDUCE(long, GA_LONG, GA_MAX, MPI_LONG, MPI_MAX, 0);
  TEST_REDUCE(long, GA_LONG, GA_MIN, MPI_LONG, MPI_MIN, 0);

  TEST_REDUCE(ulong, GA_ULONG, GA_SUM, MPI_UNSIGNED_LONG, MPI_SUM, 0);
  TEST_REDUCE(ulong, GA_ULONG, GA_PROD, MPI_UNSIGNED_LONG, MPI_PROD, 0);
  TEST_REDUCE(ulong, GA_ULONG, GA_MAX, MPI_UNSIGNED_LONG, MPI_MAX, 0);
  TEST_REDUCE(ulong, GA_ULONG, GA_MIN, MPI_UNSIGNED_LONG, MPI_MIN, 0);

  // Check failure cases
  TEST_REDUCE_FAIL(int, -1, GA_SUM, 0, GA_INVALID_ERROR);  //!< Bad data type
  TEST_REDUCE_FAIL(int, GA_INT, -1, 0, GA_INVALID_ERROR);  //!< Bad operation type
  TEST_REDUCE_FAIL(int, GA_INT, GA_SUM, SIZE - sizeof(int),
                   GA_VALUE_ERROR);  //!< Bad src offset
  // Unchecked: count upper bound, same context, dest offset (because root has
  // different behaviour than non root ranks)

  DESTROY_ARRAYS();
}
END_TEST

START_TEST(test_gpucomm_all_reduce) {}
END_TEST

START_TEST(test_gpucomm_reduce_scatter) {}
END_TEST

START_TEST(test_gpucomm_broadcast) {}
END_TEST

START_TEST(test_gpucomm_all_gather) {}
END_TEST

Suite* get_suite(void)
{
  Suite* s = suite_create("buffer_collectives");
  TCase* tc = tcase_create("API");
  tcase_add_checked_fixture(tc, setup_comm, teardown_comm);
  tcase_add_test(tc, test_gpucomm_get_count);
  tcase_add_test(tc, test_gpucomm_get_rank);
  tcase_add_test(tc, test_gpucomm_reduce);
  tcase_add_test(tc, test_gpucomm_all_reduce);
  tcase_add_test(tc, test_gpucomm_reduce_scatter);
  tcase_add_test(tc, test_gpucomm_broadcast);
  tcase_add_test(tc, test_gpucomm_all_gather);
  suite_add_tcase(s, tc);
  return s;
}
