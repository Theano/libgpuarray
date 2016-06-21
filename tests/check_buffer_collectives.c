#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include <check.h>
#include <mpi.h>

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

#define SIZE 128
#define ROOT_RANK 0
#define EPS 1.0e-9

extern gpucontext* ctx;
extern gpucomm* comm;
extern int comm_ndev;
extern int comm_rank;

extern void setup_comm(void);
extern void teardown_comm(void);

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

typedef unsigned long ulong;

/*******************************************************************************
*                Test helper buffer functions for collectives                 *
*******************************************************************************/

START_TEST(test_gpucomm_get_count) {
  int gpucount = 0, err = 0;
  err = gpucomm_get_count(comm, &gpucount);
  ck_assert_int_eq(err, GA_NO_ERROR);
  ck_assert_int_eq(gpucount, comm_ndev);
}
END_TEST

START_TEST(test_gpucomm_get_rank) {
  int rank = 0, err = 0;
  err = gpucomm_get_rank(comm, &rank);
  ck_assert_int_eq(err, GA_NO_ERROR);
  ck_assert_int_eq(rank, comm_rank);
}
END_TEST

/*******************************************************************************
*                      Test buffer collective functions                       *
*******************************************************************************/

#define INIT_ARRAYS(insize, outsize)                              \
  int err;                                                        \
  void* Av = calloc((insize), sizeof(char));                      \
  if (Av == NULL)                                                 \
    ck_abort_msg("system memory allocation failed");              \
  void* RESv = calloc((outsize), sizeof(char));                   \
  if (RESv == NULL)                                               \
    ck_abort_msg("system memory allocation failed");              \
  void* EXPv = calloc((outsize), sizeof(char));                   \
  if (EXPv == NULL)                                               \
    ck_abort_msg("system memory allocation failed");              \
  gpudata* Adev = gpudata_alloc(ctx, (insize), NULL, 0, &err);    \
  ck_assert_ptr_ne(Adev, NULL);                                   \
  gpudata* RESdev = gpudata_alloc(ctx, (outsize), NULL, 0, &err); \
  ck_assert_ptr_ne(RESdev, NULL);

#define DESTROY_ARRAYS() \
  free(Av);              \
  free(RESv);            \
  free(EXPv);            \
  gpudata_release(Adev); \
  gpudata_release(RESdev);

#define TEST_REDUCE(systype, gatype, gaoptype, mpitype, mpioptype, epsilon) \
  do {                                                                      \
    systype* A = (systype*)Av;                                              \
    systype* RES = (systype*)RESv;                                          \
    systype* EXP = (systype*)EXPv;                                          \
                                                                            \
    int i, count = SIZE / sizeof(systype);                                  \
    for (i = 0; i < count; ++i)                                             \
      A[i] = comm_rank + 2;                                                 \
    err = gpudata_write(Adev, 0, A, SIZE);                                  \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
                                                                            \
    err = gpucomm_reduce(Adev, 0, RESdev, 0, count, gatype, gaoptype,       \
                         ROOT_RANK, comm);                                  \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
    gpudata_sync(RESdev);                                                   \
    gpudata_sync(Adev);                                                     \
                                                                            \
    err = MPI_Reduce(A, EXP, count, mpitype, mpioptype, ROOT_RANK,          \
                     MPI_COMM_WORLD);                                       \
    ck_assert_msg(err == MPI_SUCCESS,                                       \
                  "openmpi error: cannot produced expected");               \
                                                                            \
    if (comm_rank == ROOT_RANK) {                                           \
      err = gpudata_read(RES, RESdev, 0, SIZE);                             \
      ck_assert_int_eq(err, GA_NO_ERROR);                                   \
      systype res;                                                          \
      MAX_ABS_DIFF(RES, EXP, count, res);                                   \
      ck_assert_msg(                                                        \
          res <= epsilon,                                                   \
          "gpudata_reduce with %s type and %s op produced max abs err %f",  \
          STR(gatype), STR(gaoptype), res);                                 \
    }                                                                       \
  } while (0)

#define TEST_REDUCE_FAIL(count, gatype, gaoptype, offsrc, experror)            \
  do {                                                                         \
    err = gpucomm_reduce(Adev, (offsrc), RESdev, 0, (count), gatype, gaoptype, \
                         ROOT_RANK, comm);                                     \
    ck_assert_int_eq(err, (experror));                                         \
    gpudata_sync(RESdev);                                                      \
    gpudata_sync(Adev);                                                        \
  } while (0)

/**
 * \note Untested for: `same context`, `dest offset`.
 * (because root has different behaviour than non root ranks)
 */
START_TEST(test_gpucomm_reduce) {
  INIT_ARRAYS(SIZE, SIZE);

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
  TEST_REDUCE_FAIL(SIZE / sizeof(int), -1, GA_SUM, 0,
                   GA_INVALID_ERROR);  //!< Bad data type
  TEST_REDUCE_FAIL(SIZE / sizeof(int), GA_INT, -1, 0,
                   GA_INVALID_ERROR);  //!< Bad operation type
  TEST_REDUCE_FAIL(SIZE / sizeof(int), GA_INT, GA_SUM, SIZE - sizeof(int),
                   GA_VALUE_ERROR);  //!< Bad src offset
  TEST_REDUCE_FAIL((size_t)INT_MAX + 1, GA_INT, GA_SUM, 0,
                   GA_UNSUPPORTED_ERROR);  //!< Too big count

  DESTROY_ARRAYS();
}
END_TEST

#define TEST_ALL_REDUCE(systype, gatype, gaoptype, mpitype, mpioptype,         \
                        epsilon)                                               \
  do {                                                                         \
    systype* A = (systype*)Av;                                                 \
    systype* RES = (systype*)RESv;                                             \
    systype* EXP = (systype*)EXPv;                                             \
                                                                               \
    int i, count = SIZE / sizeof(systype);                                     \
    for (i = 0; i < count; ++i)                                                \
      A[i] = comm_rank + 1;                                                    \
    err = gpudata_write(Adev, 0, A, SIZE);                                     \
    ck_assert_int_eq(err, GA_NO_ERROR);                                        \
                                                                               \
    err =                                                                      \
        gpucomm_all_reduce(Adev, 0, RESdev, 0, count, gatype, gaoptype, comm); \
    ck_assert_int_eq(err, GA_NO_ERROR);                                        \
    gpudata_sync(RESdev);                                                      \
    gpudata_sync(Adev);                                                        \
                                                                               \
    err = MPI_Allreduce(A, EXP, count, mpitype, mpioptype, MPI_COMM_WORLD);    \
    ck_assert_msg(err == MPI_SUCCESS,                                          \
                  "openmpi error: cannot produced expected");                  \
                                                                               \
    err = gpudata_read(RES, RESdev, 0, SIZE);                                  \
    ck_assert_int_eq(err, GA_NO_ERROR);                                        \
    systype res;                                                               \
    MAX_ABS_DIFF(RES, EXP, count, res);                                        \
    ck_assert_msg(                                                             \
        res <= epsilon,                                                        \
        "gpudata_all_reduce with %s type and %s op produced max abs err %f",   \
        STR(gatype), STR(gaoptype), res);                                      \
  } while (0)

#define TEST_ALL_REDUCE_FAIL(count, gatype, gaoptype, offsrc, offdest,   \
                             experror)                                   \
  do {                                                                   \
    err = gpucomm_all_reduce(Adev, (offsrc), RESdev, (offdest), (count), \
                             gatype, gaoptype, comm);                    \
    ck_assert_int_eq(err, (experror));                                   \
    gpudata_sync(RESdev);                                                \
    gpudata_sync(Adev);                                                  \
  } while (0)

/**
 * \note Untested for: `same context`
 */
START_TEST(test_gpucomm_all_reduce) {
  INIT_ARRAYS(SIZE, SIZE);

  // Check successful cases
  TEST_ALL_REDUCE(int, GA_INT, GA_SUM, MPI_INT, MPI_SUM, 0);
  TEST_ALL_REDUCE(int, GA_INT, GA_PROD, MPI_INT, MPI_PROD, 0);
  TEST_ALL_REDUCE(int, GA_INT, GA_MAX, MPI_INT, MPI_MAX, 0);
  TEST_ALL_REDUCE(int, GA_INT, GA_MIN, MPI_INT, MPI_MIN, 0);

  TEST_ALL_REDUCE(char, GA_BYTE, GA_SUM, MPI_BYTE, MPI_SUM, 0);
  TEST_ALL_REDUCE(char, GA_BYTE, GA_PROD, MPI_BYTE, MPI_PROD, 0);
  TEST_ALL_REDUCE(char, GA_BYTE, GA_MAX, MPI_BYTE, MPI_MAX, 0);
  TEST_ALL_REDUCE(char, GA_BYTE, GA_MIN, MPI_BYTE, MPI_MIN, 0);

  TEST_ALL_REDUCE(float, GA_FLOAT, GA_SUM, MPI_FLOAT, MPI_SUM, EPS);
  TEST_ALL_REDUCE(float, GA_FLOAT, GA_PROD, MPI_FLOAT, MPI_PROD, EPS);
  TEST_ALL_REDUCE(float, GA_FLOAT, GA_MAX, MPI_FLOAT, MPI_MAX, EPS);
  TEST_ALL_REDUCE(float, GA_FLOAT, GA_MIN, MPI_FLOAT, MPI_MIN, EPS);

  TEST_ALL_REDUCE(double, GA_DOUBLE, GA_SUM, MPI_DOUBLE, MPI_SUM, EPS);
  TEST_ALL_REDUCE(double, GA_DOUBLE, GA_PROD, MPI_DOUBLE, MPI_PROD, EPS);
  TEST_ALL_REDUCE(double, GA_DOUBLE, GA_MAX, MPI_DOUBLE, MPI_MAX, EPS);
  TEST_ALL_REDUCE(double, GA_DOUBLE, GA_MIN, MPI_DOUBLE, MPI_MIN, EPS);

  TEST_ALL_REDUCE(long, GA_LONG, GA_SUM, MPI_LONG, MPI_SUM, 0);
  TEST_ALL_REDUCE(long, GA_LONG, GA_PROD, MPI_LONG, MPI_PROD, 0);
  TEST_ALL_REDUCE(long, GA_LONG, GA_MAX, MPI_LONG, MPI_MAX, 0);
  TEST_ALL_REDUCE(long, GA_LONG, GA_MIN, MPI_LONG, MPI_MIN, 0);

  TEST_ALL_REDUCE(ulong, GA_ULONG, GA_SUM, MPI_UNSIGNED_LONG, MPI_SUM, 0);
  TEST_ALL_REDUCE(ulong, GA_ULONG, GA_PROD, MPI_UNSIGNED_LONG, MPI_PROD, 0);
  TEST_ALL_REDUCE(ulong, GA_ULONG, GA_MAX, MPI_UNSIGNED_LONG, MPI_MAX, 0);
  TEST_ALL_REDUCE(ulong, GA_ULONG, GA_MIN, MPI_UNSIGNED_LONG, MPI_MIN, 0);

  // Check failure cases
  TEST_ALL_REDUCE_FAIL(SIZE / sizeof(int), -1, GA_SUM, 0, 0,
                       GA_INVALID_ERROR);  //!< Bad data type
  TEST_ALL_REDUCE_FAIL(SIZE / sizeof(int), GA_INT, -1, 0, 0,
                       GA_INVALID_ERROR);  //!< Bad operation type
  TEST_ALL_REDUCE_FAIL(SIZE / sizeof(int), GA_INT, GA_SUM, SIZE - sizeof(int),
                       0,
                       GA_VALUE_ERROR);  //!< Bad src offset
  TEST_ALL_REDUCE_FAIL(SIZE / sizeof(int), GA_INT, GA_SUM, 0,
                       SIZE - sizeof(int),
                       GA_VALUE_ERROR);  //!< Bad dest offset
  TEST_ALL_REDUCE_FAIL((size_t)INT_MAX + 1, GA_INT, GA_SUM, 0, 0,
                       GA_UNSUPPORTED_ERROR);  //!< Too big count

  DESTROY_ARRAYS();
}
END_TEST

#define TEST_REDUCE_SCATTER(systype, gatype, gaoptype, mpitype, mpioptype,  \
                            epsilon)                                        \
  do {                                                                      \
    systype* A = (systype*)Av;                                              \
    systype* RES = (systype*)RESv;                                          \
    systype* EXP = (systype*)EXPv;                                          \
                                                                            \
    int i, count = SIZE / sizeof(systype);                                  \
    for (i = 0; i < count; ++i)                                             \
      A[i] = comm_rank + 1;                                                 \
    err = gpudata_write(Adev, 0, A, SIZE);                                  \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
                                                                            \
    int recvcount = count / comm_ndev;                                      \
    err = gpucomm_reduce_scatter(Adev, 0, RESdev, 0, recvcount, gatype,     \
                                 gaoptype, comm);                           \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
    gpudata_sync(RESdev);                                                   \
    gpudata_sync(Adev);                                                     \
                                                                            \
    int* recvcounts = (int*)malloc(comm_ndev * sizeof(int));                \
    if (recvcounts == NULL)                                                 \
      ck_abort_msg("system memory allocation failed");                      \
    for (i = 0; i < comm_ndev; ++i)                                         \
      recvcounts[i] = recvcount;                                            \
    err = MPI_Reduce_scatter(A, EXP, recvcounts, mpitype, mpioptype,        \
                             MPI_COMM_WORLD);                               \
    free(recvcounts);                                                       \
    ck_assert_msg(err == MPI_SUCCESS,                                       \
                  "openmpi error: cannot produced expected");               \
                                                                            \
    err = gpudata_read(RES, RESdev, 0, SIZE / comm_ndev);                   \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
    systype res;                                                            \
    MAX_ABS_DIFF(RES, EXP, recvcount, res);                                 \
    ck_assert_msg(res <= epsilon,                                           \
                  "gpudata_reduce_scatter with %s type and %s op produced " \
                  "max abs err %f",                                         \
                  STR(gatype), STR(gaoptype), res);                         \
  } while (0)

#define TEST_REDUCE_SCATTER_FAIL(count, gatype, gaoptype, offsrc, offdest,   \
                                 experror)                                   \
  do {                                                                       \
    err = gpucomm_reduce_scatter(Adev, (offsrc), RESdev, (offdest), (count), \
                                 gatype, gaoptype, comm);                    \
    ck_assert_int_eq(err, (experror));                                       \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
  } while (0)

/**
 * \note Untested for: `same context`
 */
START_TEST(test_gpucomm_reduce_scatter) {
  INIT_ARRAYS(SIZE, SIZE / comm_ndev);

  // Check successful cases
  TEST_REDUCE_SCATTER(int, GA_INT, GA_SUM, MPI_INT, MPI_SUM, 0);
  TEST_REDUCE_SCATTER(int, GA_INT, GA_PROD, MPI_INT, MPI_PROD, 0);
  TEST_REDUCE_SCATTER(int, GA_INT, GA_MAX, MPI_INT, MPI_MAX, 0);
  TEST_REDUCE_SCATTER(int, GA_INT, GA_MIN, MPI_INT, MPI_MIN, 0);

  TEST_REDUCE_SCATTER(char, GA_BYTE, GA_SUM, MPI_BYTE, MPI_SUM, 0);
  TEST_REDUCE_SCATTER(char, GA_BYTE, GA_PROD, MPI_BYTE, MPI_PROD, 0);
  TEST_REDUCE_SCATTER(char, GA_BYTE, GA_MAX, MPI_BYTE, MPI_MAX, 0);
  TEST_REDUCE_SCATTER(char, GA_BYTE, GA_MIN, MPI_BYTE, MPI_MIN, 0);

  TEST_REDUCE_SCATTER(float, GA_FLOAT, GA_SUM, MPI_FLOAT, MPI_SUM, EPS);
  TEST_REDUCE_SCATTER(float, GA_FLOAT, GA_PROD, MPI_FLOAT, MPI_PROD, EPS);
  TEST_REDUCE_SCATTER(float, GA_FLOAT, GA_MAX, MPI_FLOAT, MPI_MAX, EPS);
  TEST_REDUCE_SCATTER(float, GA_FLOAT, GA_MIN, MPI_FLOAT, MPI_MIN, EPS);

  TEST_REDUCE_SCATTER(double, GA_DOUBLE, GA_SUM, MPI_DOUBLE, MPI_SUM, EPS);
  TEST_REDUCE_SCATTER(double, GA_DOUBLE, GA_PROD, MPI_DOUBLE, MPI_PROD, EPS);
  TEST_REDUCE_SCATTER(double, GA_DOUBLE, GA_MAX, MPI_DOUBLE, MPI_MAX, EPS);
  TEST_REDUCE_SCATTER(double, GA_DOUBLE, GA_MIN, MPI_DOUBLE, MPI_MIN, EPS);

  TEST_REDUCE_SCATTER(long, GA_LONG, GA_SUM, MPI_LONG, MPI_SUM, 0);
  TEST_REDUCE_SCATTER(long, GA_LONG, GA_PROD, MPI_LONG, MPI_PROD, 0);
  TEST_REDUCE_SCATTER(long, GA_LONG, GA_MAX, MPI_LONG, MPI_MAX, 0);
  TEST_REDUCE_SCATTER(long, GA_LONG, GA_MIN, MPI_LONG, MPI_MIN, 0);

  TEST_REDUCE_SCATTER(ulong, GA_ULONG, GA_SUM, MPI_UNSIGNED_LONG, MPI_SUM, 0);
  TEST_REDUCE_SCATTER(ulong, GA_ULONG, GA_PROD, MPI_UNSIGNED_LONG, MPI_PROD, 0);
  TEST_REDUCE_SCATTER(ulong, GA_ULONG, GA_MAX, MPI_UNSIGNED_LONG, MPI_MAX, 0);
  TEST_REDUCE_SCATTER(ulong, GA_ULONG, GA_MIN, MPI_UNSIGNED_LONG, MPI_MIN, 0);

  // Check failure cases
  size_t outcount = SIZE / sizeof(int) / comm_ndev;
  TEST_REDUCE_SCATTER_FAIL(outcount, -1, GA_SUM, 0, 0,
                           GA_INVALID_ERROR);  //!< Bad data type
  TEST_REDUCE_SCATTER_FAIL(outcount, GA_INT, -1, 0, 0,
                           GA_INVALID_ERROR);  //!< Bad operation type
  TEST_REDUCE_SCATTER_FAIL(outcount, GA_INT, GA_SUM, SIZE - sizeof(int), 0,
                           GA_VALUE_ERROR);  //!< Bad src offset
  TEST_REDUCE_SCATTER_FAIL(outcount, GA_INT, GA_SUM, 0,
                           SIZE / comm_ndev - sizeof(int),
                           GA_VALUE_ERROR);  //!< Bad dest offset
  TEST_REDUCE_SCATTER_FAIL((size_t)INT_MAX + 1, GA_INT, GA_SUM, 0, 0,
                           GA_UNSUPPORTED_ERROR);  //!< Too big count

  DESTROY_ARRAYS();
}
END_TEST

#define TEST_BROADCAST(systype, gatype, mpitype, epsilon)                   \
  do {                                                                      \
    systype* RES = (systype*)RESv;                                          \
    systype* EXP = (systype*)EXPv;                                          \
                                                                            \
    int i, count = SIZE / sizeof(systype);                                  \
    for (i = 0; i < count; ++i) {                                           \
      RES[i] = comm_rank + 1;                                               \
      EXP[i] = RES[i];                                                      \
    }                                                                       \
    err = gpudata_write(RESdev, 0, RES, SIZE);                              \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
                                                                            \
    err = gpucomm_broadcast(RESdev, 0, count, gatype, ROOT_RANK, comm);     \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
    gpudata_sync(RESdev);                                                   \
                                                                            \
    err = MPI_Bcast(EXP, count, mpitype, ROOT_RANK, MPI_COMM_WORLD);        \
    ck_assert_msg(err == MPI_SUCCESS,                                       \
                  "openmpi error: cannot produced expected");               \
                                                                            \
    err = gpudata_read(RES, RESdev, 0, SIZE);                               \
    ck_assert_int_eq(err, GA_NO_ERROR);                                     \
    systype res;                                                            \
    MAX_ABS_DIFF(RES, EXP, count, res);                                     \
    ck_assert_msg(res <= epsilon,                                           \
                  "gpudata_broadcast with %s type produced max abs err %f", \
                  STR(gatype), res);                                        \
  } while (0)

#define TEST_BROADCAST_FAIL(count, gatype, offsrc, experror)                   \
  do {                                                                         \
    err =                                                                      \
        gpucomm_broadcast(RESdev, (offsrc), (count), gatype, ROOT_RANK, comm); \
    ck_assert_int_eq(err, (experror));                                         \
    gpudata_sync(RESdev);                                                      \
  } while (0)

/**
 * \note Untested for: `same context`
 */
START_TEST(test_gpucomm_broadcast) {
  INIT_ARRAYS(SIZE, SIZE);

  // Check successful cases
  TEST_BROADCAST(int, GA_INT, MPI_INT, 0);
  TEST_BROADCAST(char, GA_BYTE, MPI_BYTE, 0);
  TEST_BROADCAST(float, GA_FLOAT, MPI_FLOAT, EPS);
  TEST_BROADCAST(double, GA_DOUBLE, MPI_DOUBLE, EPS);
  TEST_BROADCAST(long, GA_LONG, MPI_LONG, 0);
  TEST_BROADCAST(ulong, GA_ULONG, MPI_UNSIGNED_LONG, 0);

  // Check failure cases
  TEST_BROADCAST_FAIL(SIZE / sizeof(int), -1, 0,
                      GA_INVALID_ERROR);  //!< Bad data type
  TEST_BROADCAST_FAIL(SIZE / sizeof(int), GA_INT, SIZE - sizeof(int),
                      GA_VALUE_ERROR);  //!< Bad src offset
  TEST_BROADCAST_FAIL((size_t)INT_MAX + 1, GA_INT, 0,
                      GA_UNSUPPORTED_ERROR);  //!< Too big count

  DESTROY_ARRAYS();
}
END_TEST

#define TEST_ALL_GATHER(systype, gatype, mpitype, epsilon)                   \
  do {                                                                       \
    systype* A = (systype*)Av;                                               \
    systype* RES = (systype*)RESv;                                           \
    systype* EXP = (systype*)EXPv;                                           \
                                                                             \
    int i, count = SIZE / sizeof(systype);                                   \
    int sendcount = count / comm_ndev;                                       \
    for (i = 0; i < sendcount; ++i)                                          \
      A[i] = comm_rank + 1;                                                  \
    err = gpudata_write(Adev, 0, A, SIZE / comm_ndev);                       \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
                                                                             \
    err = gpucomm_all_gather(Adev, 0, RESdev, 0, sendcount, gatype, comm);   \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
                                                                             \
    err = MPI_Allgather(A, sendcount, mpitype, EXP, count, mpitype,          \
                        MPI_COMM_WORLD);                                     \
    ck_assert_msg(err == MPI_SUCCESS,                                        \
                  "openmpi error: cannot produced expected");                \
                                                                             \
    err = gpudata_read(RES, RESdev, 0, SIZE);                                \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    systype res;                                                             \
    MAX_ABS_DIFF(RES, EXP, count, res);                                      \
    ck_assert_msg(res <= epsilon,                                            \
                  "gpudata_all_gather with %s type produced max abs err %f", \
                  STR(gatype), res);                                         \
  } while (0)

#define TEST_ALL_GATHER_FAIL(count, gatype, offsrc, offdest, experror)   \
  do {                                                                   \
    err = gpucomm_all_gather(Adev, (offsrc), RESdev, (offdest), (count), \
                             gatype, comm);                              \
    ck_assert_int_eq(err, (experror));                                   \
    gpudata_sync(RESdev);                                                \
    gpudata_sync(Adev);                                                  \
  } while (0)

/**
 * \note Untested for: `same context`
 */
START_TEST(test_gpucomm_all_gather) {
  INIT_ARRAYS(SIZE / comm_ndev, SIZE);

  // Check successful cases
  TEST_ALL_GATHER(int, GA_INT, MPI_INT, 0);
  TEST_ALL_GATHER(char, GA_BYTE, MPI_BYTE, 0);
  TEST_ALL_GATHER(float, GA_FLOAT, MPI_FLOAT, EPS);
  TEST_ALL_GATHER(double, GA_DOUBLE, MPI_DOUBLE, EPS);
  TEST_ALL_GATHER(long, GA_LONG, MPI_LONG, 0);
  TEST_ALL_GATHER(ulong, GA_ULONG, MPI_UNSIGNED_LONG, 0);

  // Check failure cases
  int incount = SIZE / sizeof(int) / comm_ndev;
  TEST_ALL_GATHER_FAIL(incount, -1, 0, 0, GA_INVALID_ERROR);  //!< Bad data type
  TEST_ALL_GATHER_FAIL(incount, GA_INT, SIZE / comm_ndev - sizeof(int), 0,
                       GA_VALUE_ERROR);  //!< Bad src offset
  TEST_ALL_GATHER_FAIL(incount, GA_INT, 0, SIZE - sizeof(int),
                       GA_VALUE_ERROR);  //!< Bad dest offset
  TEST_ALL_GATHER_FAIL((size_t)INT_MAX + 1, GA_INT, 0, 0,
                       GA_UNSUPPORTED_ERROR);  //!< Too big count

  DESTROY_ARRAYS();
}
END_TEST

Suite* get_suite(void) {
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
