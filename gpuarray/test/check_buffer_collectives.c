#include <limits.h>
#include <math.h>
#include <stdio.h>
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

#define PRINTV(ar, N, t)           \
  do {                             \
    printf("%s\n", STR(ar));       \
    int li;                        \
    for (li = 0; li < (N); ++li) { \
      printf(STR(t) " ", ar[li]);  \
    }                              \
    printf("\n");                  \
    printf("\n");                  \
  } while (0)
#define PRINTVF(ar, N) PRINTV(ar, N, %.2f)
#define PRINTVI(ar, N) PRINTV(ar, N, %i)
#define PRINTVL(ar, N) PRINTV(ar, N, %li)
#define PRINTVUL(ar, N) PRINTV(ar, N, %lu)

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

#define TEST_REDUCE(systype, gatype, mpitype, coloptype, epsilon, print)       \
  START_TEST(test_gpucomm_reduce_##gatype##_##coloptype) {                     \
    INIT_ARRAYS(SIZE, SIZE)                                                    \
                                                                               \
    systype* A = (systype*)Av;                                                 \
    systype* RES = (systype*)RESv;                                             \
    systype* EXP = (systype*)EXPv;                                             \
                                                                               \
    int i, count = SIZE / sizeof(systype);                                     \
    for (i = 0; i < count; ++i)                                                \
      A[i] = comm_rank + 2;                                                    \
    err = gpudata_write(Adev, 0, A, SIZE);                                     \
    ck_assert_int_eq(err, GA_NO_ERROR);                                        \
                                                                               \
    err = gpucomm_reduce(Adev, 0, RESdev, 0, count, GA_##gatype,               \
                         GA_##coloptype, ROOT_RANK, comm);                     \
    ck_assert_int_eq(err, GA_NO_ERROR);                                        \
    gpudata_sync(RESdev);                                                      \
    gpudata_sync(Adev);                                                        \
                                                                               \
    err = MPI_Reduce(A, EXP, count, MPI_##mpitype, MPI_##coloptype, ROOT_RANK, \
                     MPI_COMM_WORLD);                                          \
    ck_assert_msg(err == MPI_SUCCESS,                                          \
                  "openmpi error: cannot produced expected");                  \
                                                                               \
    if (comm_rank == ROOT_RANK) {                                              \
      err = gpudata_read(RES, RESdev, 0, SIZE);                                \
      ck_assert_int_eq(err, GA_NO_ERROR);                                      \
      systype res;                                                             \
      MAX_ABS_DIFF(RES, EXP, count, res);                                      \
      if (!(res <= epsilon)) {                                                 \
        print(RES, count);                                                     \
        print(EXP, count);                                                     \
        ck_abort_msg(                                                          \
            "gpudata_reduce with %s type and %s op produced max abs err %.1f", \
            STR(GA_##gatype), STR(GA_##coloptype), (double)res);               \
      }                                                                        \
    }                                                                          \
                                                                               \
    DESTROY_ARRAYS()                                                           \
  }                                                                            \
  END_TEST

#define TEST_REDUCE_FAIL(tname, count, gatype, gaoptype, offsrc, experror)     \
  START_TEST(test_gpucomm_reduce_fail_##tname) {                               \
    INIT_ARRAYS(SIZE, SIZE)                                                    \
    err = gpucomm_reduce(Adev, (offsrc), RESdev, 0, (count), gatype, gaoptype, \
                         ROOT_RANK, comm);                                     \
    ck_assert_int_eq(err, (experror));                                         \
    gpudata_sync(RESdev);                                                      \
    gpudata_sync(Adev);                                                        \
    DESTROY_ARRAYS()                                                           \
  }                                                                            \
  END_TEST

/**
 * \note Untested for: half datatype, `same context`, `dest offset`.
 * (because root has different behaviour than non root ranks)
 */
// Success tests
TEST_REDUCE(int, INT, INT, SUM, 0, PRINTVI)
TEST_REDUCE(int, INT, INT, PROD, 0, PRINTVI)
TEST_REDUCE(int, INT, INT, MAX, 0, PRINTVI)
TEST_REDUCE(int, INT, INT, MIN, 0, PRINTVI)
TEST_REDUCE(char, BYTE, BYTE, SUM, 0, PRINTVI)
TEST_REDUCE(char, BYTE, BYTE, PROD, 0, PRINTVI)
TEST_REDUCE(char, BYTE, BYTE, MAX, 0, PRINTVI)
TEST_REDUCE(char, BYTE, BYTE, MIN, 0, PRINTVI)
TEST_REDUCE(float, FLOAT, FLOAT, SUM, EPS, PRINTVF)
TEST_REDUCE(float, FLOAT, FLOAT, PROD, EPS, PRINTVF)
TEST_REDUCE(float, FLOAT, FLOAT, MAX, EPS, PRINTVF)
TEST_REDUCE(float, FLOAT, FLOAT, MIN, EPS, PRINTVF)
TEST_REDUCE(double, DOUBLE, DOUBLE, SUM, EPS, PRINTVF)
TEST_REDUCE(double, DOUBLE, DOUBLE, PROD, EPS, PRINTVF)
TEST_REDUCE(double, DOUBLE, DOUBLE, MAX, EPS, PRINTVF)
TEST_REDUCE(double, DOUBLE, DOUBLE, MIN, EPS, PRINTVF)
TEST_REDUCE(long, LONG, LONG, SUM, 0, PRINTVL)
TEST_REDUCE(long, LONG, LONG, PROD, 0, PRINTVL)
TEST_REDUCE(long, LONG, LONG, MAX, 0, PRINTVL)
TEST_REDUCE(long, LONG, LONG, MIN, 0, PRINTVL)
TEST_REDUCE(ulong, ULONG, UNSIGNED_LONG, SUM, 0, PRINTVUL)
TEST_REDUCE(ulong, ULONG, UNSIGNED_LONG, PROD, 0, PRINTVUL)
TEST_REDUCE(ulong, ULONG, UNSIGNED_LONG, MAX, 0, PRINTVUL)
TEST_REDUCE(ulong, ULONG, UNSIGNED_LONG, MIN, 0, PRINTVUL)

// Failure tests
TEST_REDUCE_FAIL(datatype, SIZE / sizeof(int), -1, GA_SUM, 0, GA_INVALID_ERROR)
TEST_REDUCE_FAIL(optype, SIZE / sizeof(int), GA_INT, -1, 0, GA_INVALID_ERROR)
TEST_REDUCE_FAIL(src_offset, SIZE / sizeof(int), GA_INT, GA_SUM,
                 SIZE - sizeof(int), GA_VALUE_ERROR)
TEST_REDUCE_FAIL(elemcount, (size_t)INT_MAX + 1, GA_INT, GA_SUM, 0,
                 GA_UNSUPPORTED_ERROR)

#define TEST_ALL_REDUCE(systype, gatype, mpitype, coloptype, epsilon, print) \
  START_TEST(test_gpucomm_all_reduce_##gatype##_##coloptype) {               \
    INIT_ARRAYS(SIZE, SIZE)                                                  \
                                                                             \
    systype* A = (systype*)Av;                                               \
    systype* RES = (systype*)RESv;                                           \
    systype* EXP = (systype*)EXPv;                                           \
                                                                             \
    int i, count = SIZE / sizeof(systype);                                   \
    for (i = 0; i < count; ++i)                                              \
      A[i] = comm_rank + 2;                                                  \
    err = gpudata_write(Adev, 0, A, SIZE);                                   \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
                                                                             \
    err = gpucomm_all_reduce(Adev, 0, RESdev, 0, count, GA_##gatype,         \
                             GA_##coloptype, comm);                          \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
                                                                             \
    err = MPI_Allreduce(A, EXP, count, MPI_##mpitype, MPI_##coloptype,       \
                        MPI_COMM_WORLD);                                     \
    ck_assert_msg(err == MPI_SUCCESS,                                        \
                  "openmpi error: cannot produced expected");                \
                                                                             \
    err = gpudata_read(RES, RESdev, 0, SIZE);                                \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    systype res;                                                             \
    MAX_ABS_DIFF(RES, EXP, count, res);                                      \
    if (!(res <= epsilon)) {                                                 \
      print(RES, count);                                                     \
      print(EXP, count);                                                     \
      ck_abort_msg(                                                          \
          "gpudata_all_reduce with %s type and %s op produced max abs err "  \
          "%.1f",                                                            \
          STR(GA_##gatype), STR(GA_##coloptype), (double)res);               \
    }                                                                        \
                                                                             \
    DESTROY_ARRAYS()                                                         \
  }                                                                          \
  END_TEST

#define TEST_ALL_REDUCE_FAIL(tname, count, gatype, gaoptype, offsrc, offdest, \
                             experror)                                        \
  START_TEST(test_gpucomm_all_reduce_fail_##tname) {                          \
    INIT_ARRAYS(SIZE, SIZE)                                                   \
    err = gpucomm_all_reduce(Adev, (offsrc), RESdev, (offdest), (count),      \
                             gatype, gaoptype, comm);                         \
    ck_assert_int_eq(err, (experror));                                        \
    gpudata_sync(RESdev);                                                     \
    gpudata_sync(Adev);                                                       \
    DESTROY_ARRAYS()                                                          \
  }                                                                           \
  END_TEST

/**
 * \note Untested for: half datatype, `same context`
 */
// Success tests
TEST_ALL_REDUCE(int, INT, INT, SUM, 0, PRINTVI)
TEST_ALL_REDUCE(int, INT, INT, PROD, 0, PRINTVI)
TEST_ALL_REDUCE(int, INT, INT, MAX, 0, PRINTVI)
TEST_ALL_REDUCE(int, INT, INT, MIN, 0, PRINTVI)
TEST_ALL_REDUCE(char, BYTE, BYTE, SUM, 0, PRINTVI)
TEST_ALL_REDUCE(char, BYTE, BYTE, PROD, 0, PRINTVI)
TEST_ALL_REDUCE(char, BYTE, BYTE, MAX, 0, PRINTVI)
TEST_ALL_REDUCE(char, BYTE, BYTE, MIN, 0, PRINTVI)
TEST_ALL_REDUCE(float, FLOAT, FLOAT, SUM, EPS, PRINTVF)
TEST_ALL_REDUCE(float, FLOAT, FLOAT, PROD, EPS, PRINTVF)
TEST_ALL_REDUCE(float, FLOAT, FLOAT, MAX, EPS, PRINTVF)
TEST_ALL_REDUCE(float, FLOAT, FLOAT, MIN, EPS, PRINTVF)
TEST_ALL_REDUCE(double, DOUBLE, DOUBLE, SUM, EPS, PRINTVF)
TEST_ALL_REDUCE(double, DOUBLE, DOUBLE, PROD, EPS, PRINTVF)
TEST_ALL_REDUCE(double, DOUBLE, DOUBLE, MAX, EPS, PRINTVF)
TEST_ALL_REDUCE(double, DOUBLE, DOUBLE, MIN, EPS, PRINTVF)
TEST_ALL_REDUCE(long, LONG, LONG, SUM, 0, PRINTVL)
TEST_ALL_REDUCE(long, LONG, LONG, PROD, 0, PRINTVL)
TEST_ALL_REDUCE(long, LONG, LONG, MAX, 0, PRINTVL)
TEST_ALL_REDUCE(long, LONG, LONG, MIN, 0, PRINTVL)
TEST_ALL_REDUCE(ulong, ULONG, UNSIGNED_LONG, SUM, 0, PRINTVUL)
TEST_ALL_REDUCE(ulong, ULONG, UNSIGNED_LONG, PROD, 0, PRINTVUL)
TEST_ALL_REDUCE(ulong, ULONG, UNSIGNED_LONG, MAX, 0, PRINTVUL)
TEST_ALL_REDUCE(ulong, ULONG, UNSIGNED_LONG, MIN, 0, PRINTVUL)

// Failure tests
TEST_ALL_REDUCE_FAIL(datatype, SIZE / sizeof(int), -1, GA_SUM, 0, 0,
                     GA_INVALID_ERROR)
TEST_ALL_REDUCE_FAIL(optype, SIZE / sizeof(int), GA_INT, -1, 0, 0,
                     GA_INVALID_ERROR)
TEST_ALL_REDUCE_FAIL(src_offset, SIZE / sizeof(int), GA_INT, GA_SUM,
                     SIZE - sizeof(int), 0, GA_VALUE_ERROR)
TEST_ALL_REDUCE_FAIL(dest_offset, SIZE / sizeof(int), GA_INT, GA_SUM, 0,
                     SIZE - sizeof(int), GA_VALUE_ERROR)
TEST_ALL_REDUCE_FAIL(elemcount, (size_t)INT_MAX + 1, GA_INT, GA_SUM, 0, 0,
                     GA_UNSUPPORTED_ERROR)

#define TEST_REDUCE_SCATTER(systype, gatype, mpitype, coloptype, epsilon,    \
                            print)                                           \
  START_TEST(test_gpucomm_reduce_scatter_##gatype##_##coloptype) {           \
    INIT_ARRAYS(SIZE, SIZE / comm_ndev)                                      \
                                                                             \
    systype* A = (systype*)Av;                                               \
    systype* RES = (systype*)RESv;                                           \
    systype* EXP = (systype*)EXPv;                                           \
                                                                             \
    int i, count = SIZE / sizeof(systype);                                   \
    for (i = 0; i < count; ++i)                                              \
      A[i] = comm_rank + 2;                                                  \
    err = gpudata_write(Adev, 0, A, SIZE);                                   \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
                                                                             \
    int recvcount = count / comm_ndev;                                       \
    err = gpucomm_reduce_scatter(Adev, 0, RESdev, 0, recvcount, GA_##gatype, \
                                 GA_##coloptype, comm);                      \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
                                                                             \
    int* recvcounts = (int*)malloc(comm_ndev * sizeof(int));                 \
    if (recvcounts == NULL)                                                  \
      ck_abort_msg("system memory allocation failed");                       \
    for (i = 0; i < comm_ndev; ++i)                                          \
      recvcounts[i] = recvcount;                                             \
    err = MPI_Reduce_scatter(A, EXP, recvcounts, MPI_##mpitype,              \
                             MPI_##coloptype, MPI_COMM_WORLD);               \
    free(recvcounts);                                                        \
    ck_assert_msg(err == MPI_SUCCESS,                                        \
                  "openmpi error: cannot produced expected");                \
                                                                             \
    err = gpudata_read(RES, RESdev, 0, SIZE / comm_ndev);                    \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    systype res;                                                             \
    MAX_ABS_DIFF(RES, EXP, recvcount, res);                                  \
    if (!(res <= epsilon)) {                                                 \
      print(RES, recvcount);                                                 \
      print(EXP, recvcount);                                                 \
      ck_abort_msg(                                                          \
          "gpudata_reduce_scatter with %s type and %s op produced "          \
          "max abs err %f",                                                  \
          STR(GA_##gatype), STR(GA_##coloptype), (double)res);               \
    }                                                                        \
                                                                             \
    DESTROY_ARRAYS()                                                         \
  }                                                                          \
  END_TEST

#define TEST_REDUCE_SCATTER_FAIL(tname, count, gatype, gaoptype, offsrc,     \
                                 offdest, experror)                          \
  START_TEST(test_gpucomm_reduce_scatter_fail_##tname) {                     \
    INIT_ARRAYS(SIZE, SIZE / comm_ndev)                                      \
    err = gpucomm_reduce_scatter(Adev, (offsrc), RESdev, (offdest), (count), \
                                 gatype, gaoptype, comm);                    \
    ck_assert_int_eq(err, (experror));                                       \
    gpudata_sync(RESdev);                                                    \
    gpudata_sync(Adev);                                                      \
    DESTROY_ARRAYS()                                                         \
  }                                                                          \
  END_TEST

/**
 * \note Untested for: half datatype, `same context`
 */
// Success tests
TEST_REDUCE_SCATTER(int, INT, INT, SUM, 0, PRINTVI)
TEST_REDUCE_SCATTER(int, INT, INT, PROD, 0, PRINTVI)
TEST_REDUCE_SCATTER(int, INT, INT, MAX, 0, PRINTVI)
TEST_REDUCE_SCATTER(int, INT, INT, MIN, 0, PRINTVI)
TEST_REDUCE_SCATTER(char, BYTE, BYTE, SUM, 0, PRINTVI)
TEST_REDUCE_SCATTER(char, BYTE, BYTE, PROD, 0, PRINTVI)
TEST_REDUCE_SCATTER(char, BYTE, BYTE, MAX, 0, PRINTVI)
TEST_REDUCE_SCATTER(char, BYTE, BYTE, MIN, 0, PRINTVI)
TEST_REDUCE_SCATTER(float, FLOAT, FLOAT, SUM, EPS, PRINTVF)
TEST_REDUCE_SCATTER(float, FLOAT, FLOAT, PROD, EPS, PRINTVF)
TEST_REDUCE_SCATTER(float, FLOAT, FLOAT, MAX, EPS, PRINTVF)
TEST_REDUCE_SCATTER(float, FLOAT, FLOAT, MIN, EPS, PRINTVF)
TEST_REDUCE_SCATTER(double, DOUBLE, DOUBLE, SUM, EPS, PRINTVF)
TEST_REDUCE_SCATTER(double, DOUBLE, DOUBLE, PROD, EPS, PRINTVF)
TEST_REDUCE_SCATTER(double, DOUBLE, DOUBLE, MAX, EPS, PRINTVF)
TEST_REDUCE_SCATTER(double, DOUBLE, DOUBLE, MIN, EPS, PRINTVF)
TEST_REDUCE_SCATTER(long, LONG, LONG, SUM, 0, PRINTVL)
TEST_REDUCE_SCATTER(long, LONG, LONG, PROD, 0, PRINTVL)
TEST_REDUCE_SCATTER(long, LONG, LONG, MAX, 0, PRINTVL)
TEST_REDUCE_SCATTER(long, LONG, LONG, MIN, 0, PRINTVL)
TEST_REDUCE_SCATTER(ulong, ULONG, UNSIGNED_LONG, SUM, 0, PRINTVUL)
TEST_REDUCE_SCATTER(ulong, ULONG, UNSIGNED_LONG, PROD, 0, PRINTVUL)
TEST_REDUCE_SCATTER(ulong, ULONG, UNSIGNED_LONG, MAX, 0, PRINTVUL)
TEST_REDUCE_SCATTER(ulong, ULONG, UNSIGNED_LONG, MIN, 0, PRINTVUL)

// Failure tests
#define outcount SIZE / sizeof(int) / comm_ndev
TEST_REDUCE_SCATTER_FAIL(datatype, outcount, -1, GA_SUM, 0, 0, GA_INVALID_ERROR)
TEST_REDUCE_SCATTER_FAIL(optype, outcount, GA_INT, -1, 0, 0, GA_INVALID_ERROR)
TEST_REDUCE_SCATTER_FAIL(src_offset, outcount, GA_INT, GA_SUM,
                         SIZE - sizeof(int), 0, GA_VALUE_ERROR)
TEST_REDUCE_SCATTER_FAIL(dest_offset, outcount, GA_INT, GA_SUM, 0,
                         SIZE / comm_ndev - sizeof(int), GA_VALUE_ERROR)
TEST_REDUCE_SCATTER_FAIL(elemcount, (size_t)INT_MAX + 1, GA_INT, GA_SUM, 0, 0,
                         GA_UNSUPPORTED_ERROR)

#define TEST_BROADCAST(systype, gatype, mpitype, epsilon, print)             \
  START_TEST(test_gpucomm_broadcast_##gatype) {                              \
    INIT_ARRAYS(SIZE, SIZE)                                                  \
                                                                             \
    systype* RES = (systype*)RESv;                                           \
    systype* EXP = (systype*)EXPv;                                           \
                                                                             \
    int i, count = SIZE / sizeof(systype);                                   \
    for (i = 0; i < count; ++i) {                                            \
      RES[i] = comm_rank + 1;                                                \
      EXP[i] = RES[i];                                                       \
    }                                                                        \
    err = gpudata_write(RESdev, 0, RES, SIZE);                               \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
                                                                             \
    err = gpucomm_broadcast(RESdev, 0, count, GA_##gatype, ROOT_RANK, comm); \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    gpudata_sync(RESdev);                                                    \
                                                                             \
    err = MPI_Bcast(EXP, count, MPI_##mpitype, ROOT_RANK, MPI_COMM_WORLD);   \
    ck_assert_msg(err == MPI_SUCCESS,                                        \
                  "openmpi error: cannot produced expected");                \
                                                                             \
    err = gpudata_read(RES, RESdev, 0, SIZE);                                \
    ck_assert_int_eq(err, GA_NO_ERROR);                                      \
    systype res;                                                             \
    MAX_ABS_DIFF(RES, EXP, count, res);                                      \
    if (!(res <= epsilon)) {                                                 \
      print(RES, count);                                                     \
      print(EXP, count);                                                     \
      ck_abort_msg("gpudata_broadcast with %s type produced max abs err %f", \
                   STR(GA_##gatype), (double)res);                           \
    }                                                                        \
                                                                             \
    DESTROY_ARRAYS()                                                         \
  }                                                                          \
  END_TEST

#define TEST_BROADCAST_FAIL(tname, count, gatype, offsrc, experror)            \
  START_TEST(test_gpucomm_broadcast_fail_##tname) {                            \
    INIT_ARRAYS(SIZE, SIZE)                                                    \
    err =                                                                      \
        gpucomm_broadcast(RESdev, (offsrc), (count), gatype, ROOT_RANK, comm); \
    ck_assert_int_eq(err, (experror));                                         \
    gpudata_sync(RESdev);                                                      \
    DESTROY_ARRAYS()                                                           \
  }                                                                            \
  END_TEST

/**
 * \note Untested for: half datatype, `same context`
 */
// Success tests
TEST_BROADCAST(int, INT, INT, 0, PRINTVI)
TEST_BROADCAST(char, BYTE, BYTE, 0, PRINTVI)
TEST_BROADCAST(float, FLOAT, FLOAT, EPS, PRINTVF)
TEST_BROADCAST(double, DOUBLE, DOUBLE, EPS, PRINTVF)
TEST_BROADCAST(long, LONG, LONG, 0, PRINTVL)
TEST_BROADCAST(ulong, ULONG, UNSIGNED_LONG, 0, PRINTVUL)

// Failure tests
TEST_BROADCAST_FAIL(datatype, SIZE / sizeof(int), -1, 0, GA_INVALID_ERROR)
TEST_BROADCAST_FAIL(src_offset, SIZE / sizeof(int), GA_INT, SIZE - sizeof(int),
                    GA_VALUE_ERROR)
TEST_BROADCAST_FAIL(elemcount, (size_t)INT_MAX + 1, GA_INT, 0,
                    GA_UNSUPPORTED_ERROR)

#define TEST_ALL_GATHER(systype, gatype, mpitype, epsilon, print)             \
  START_TEST(test_gpucomm_all_gather_##gatype) {                              \
    INIT_ARRAYS(SIZE / comm_ndev, SIZE)                                       \
                                                                              \
    systype* A = (systype*)Av;                                                \
    systype* RES = (systype*)RESv;                                            \
    systype* EXP = (systype*)EXPv;                                            \
                                                                              \
    int i, count = SIZE / sizeof(systype);                                    \
    int sendcount = count / comm_ndev;                                        \
    for (i = 0; i < sendcount; ++i)                                           \
      A[i] = comm_rank + 1;                                                   \
    err = gpudata_write(Adev, 0, A, SIZE / comm_ndev);                        \
    ck_assert_int_eq(err, GA_NO_ERROR);                                       \
                                                                              \
    err =                                                                     \
        gpucomm_all_gather(Adev, 0, RESdev, 0, sendcount, GA_##gatype, comm); \
    ck_assert_int_eq(err, GA_NO_ERROR);                                       \
    gpudata_sync(RESdev);                                                     \
    gpudata_sync(Adev);                                                       \
                                                                              \
    err = MPI_Allgather(A, sendcount, MPI_##mpitype, EXP, sendcount,          \
                        MPI_##mpitype, MPI_COMM_WORLD);                       \
    ck_assert_msg(err == MPI_SUCCESS,                                         \
                  "openmpi error: cannot produced expected");                 \
                                                                              \
    err = gpudata_read(RES, RESdev, 0, SIZE);                                 \
    ck_assert_int_eq(err, GA_NO_ERROR);                                       \
    systype res;                                                              \
    MAX_ABS_DIFF(RES, EXP, count, res);                                       \
    if (!(res <= epsilon)) {                                                  \
      print(RES, count);                                                      \
      print(EXP, count);                                                      \
      ck_abort_msg("gpudata_all_gather with %s type produced max abs err %f", \
                   STR(GA_##gatype), (double)res);                            \
    }                                                                         \
                                                                              \
    DESTROY_ARRAYS()                                                          \
  }                                                                           \
  END_TEST

#define TEST_ALL_GATHER_FAIL(tname, count, gatype, offsrc, offdest, experror) \
  START_TEST(test_gpucomm_all_gather_fail_##tname) {                          \
    INIT_ARRAYS(SIZE / comm_ndev, SIZE)                                       \
    err = gpucomm_all_gather(Adev, (offsrc), RESdev, (offdest), (count),      \
                             gatype, comm);                                   \
    ck_assert_int_eq(err, (experror));                                        \
    gpudata_sync(RESdev);                                                     \
    gpudata_sync(Adev);                                                       \
    DESTROY_ARRAYS()                                                          \
  }                                                                           \
  END_TEST

/**
 * \note Untested for: half datatype, `same context`
 */
// Success tests
TEST_ALL_GATHER(int, INT, INT, 0, PRINTVI)
TEST_ALL_GATHER(char, BYTE, BYTE, 0, PRINTVI)
TEST_ALL_GATHER(float, FLOAT, FLOAT, EPS, PRINTVF)
TEST_ALL_GATHER(double, DOUBLE, DOUBLE, EPS, PRINTVF)
TEST_ALL_GATHER(long, LONG, LONG, 0, PRINTVL)
TEST_ALL_GATHER(ulong, ULONG, UNSIGNED_LONG, 0, PRINTVUL)

// Failure tests
#define incount SIZE / sizeof(int) / comm_ndev
TEST_ALL_GATHER_FAIL(datatype, incount, -1, 0, 0, GA_INVALID_ERROR)
TEST_ALL_GATHER_FAIL(src_offset, incount, GA_INT,
                     SIZE / comm_ndev - sizeof(int), 0, GA_VALUE_ERROR)
TEST_ALL_GATHER_FAIL(dest_offset, incount, GA_INT, 0, SIZE - sizeof(int),
                     GA_VALUE_ERROR)
TEST_ALL_GATHER_FAIL(elemcount, (size_t)INT_MAX + 1, GA_INT, 0, 0,
                     GA_UNSUPPORTED_ERROR)

Suite* get_suite(void) {
  Suite* s = suite_create("buffer_collectives_API");

  TCase* helps = tcase_create("test_helpers");
  tcase_add_unchecked_fixture(helps, setup_comm, teardown_comm);
  tcase_add_test(helps, test_gpucomm_get_count);
  tcase_add_test(helps, test_gpucomm_get_rank);

  TCase* reds = tcase_create("test_reduce");
  tcase_add_unchecked_fixture(reds, setup_comm, teardown_comm);
  tcase_add_test(reds, test_gpucomm_reduce_INT_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_INT_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_INT_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_INT_MIN);
  tcase_add_test(reds, test_gpucomm_reduce_BYTE_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_BYTE_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_BYTE_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_BYTE_MIN);
  tcase_add_test(reds, test_gpucomm_reduce_FLOAT_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_FLOAT_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_FLOAT_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_FLOAT_MIN);
  tcase_add_test(reds, test_gpucomm_reduce_DOUBLE_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_DOUBLE_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_DOUBLE_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_DOUBLE_MIN);
  tcase_add_test(reds, test_gpucomm_reduce_LONG_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_LONG_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_LONG_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_LONG_MIN);
  tcase_add_test(reds, test_gpucomm_reduce_ULONG_SUM);
  tcase_add_test(reds, test_gpucomm_reduce_ULONG_PROD);
  tcase_add_test(reds, test_gpucomm_reduce_ULONG_MAX);
  tcase_add_test(reds, test_gpucomm_reduce_ULONG_MIN);

  TCase* redf = tcase_create("test_reduce_fail");
  tcase_add_unchecked_fixture(redf, setup_comm, teardown_comm);
  tcase_add_test(redf, test_gpucomm_reduce_fail_datatype);
  tcase_add_test(redf, test_gpucomm_reduce_fail_optype);
  tcase_add_test(redf, test_gpucomm_reduce_fail_src_offset);
  tcase_add_test(redf, test_gpucomm_reduce_fail_elemcount);

  TCase* areds = tcase_create("test_all_reduce");
  tcase_add_unchecked_fixture(areds, setup_comm, teardown_comm);
  tcase_add_test(areds, test_gpucomm_all_reduce_INT_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_INT_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_INT_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_INT_MIN);
  tcase_add_test(areds, test_gpucomm_all_reduce_BYTE_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_BYTE_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_BYTE_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_BYTE_MIN);
  tcase_add_test(areds, test_gpucomm_all_reduce_FLOAT_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_FLOAT_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_FLOAT_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_FLOAT_MIN);
  tcase_add_test(areds, test_gpucomm_all_reduce_DOUBLE_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_DOUBLE_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_DOUBLE_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_DOUBLE_MIN);
  tcase_add_test(areds, test_gpucomm_all_reduce_LONG_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_LONG_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_LONG_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_LONG_MIN);
  tcase_add_test(areds, test_gpucomm_all_reduce_ULONG_SUM);
  tcase_add_test(areds, test_gpucomm_all_reduce_ULONG_PROD);
  tcase_add_test(areds, test_gpucomm_all_reduce_ULONG_MAX);
  tcase_add_test(areds, test_gpucomm_all_reduce_ULONG_MIN);

  TCase* aredf = tcase_create("test_all_reduce_fail");
  tcase_add_unchecked_fixture(aredf, setup_comm, teardown_comm);
  tcase_add_test(aredf, test_gpucomm_all_reduce_fail_datatype);
  tcase_add_test(aredf, test_gpucomm_all_reduce_fail_optype);
  tcase_add_test(aredf, test_gpucomm_all_reduce_fail_src_offset);
  tcase_add_test(aredf, test_gpucomm_all_reduce_fail_dest_offset);
  tcase_add_test(aredf, test_gpucomm_all_reduce_fail_elemcount);

  TCase* redscs = tcase_create("test_reduce_scatter");
  tcase_add_unchecked_fixture(redscs, setup_comm, teardown_comm);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_INT_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_INT_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_INT_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_INT_MIN);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_BYTE_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_BYTE_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_BYTE_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_BYTE_MIN);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_FLOAT_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_FLOAT_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_FLOAT_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_FLOAT_MIN);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_DOUBLE_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_DOUBLE_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_DOUBLE_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_DOUBLE_MIN);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_LONG_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_LONG_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_LONG_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_LONG_MIN);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_ULONG_SUM);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_ULONG_PROD);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_ULONG_MAX);
  tcase_add_test(redscs, test_gpucomm_reduce_scatter_ULONG_MIN);

  TCase* redscf = tcase_create("test_reduce_scatter_fail");
  tcase_add_unchecked_fixture(redscf, setup_comm, teardown_comm);
  tcase_add_test(redscf, test_gpucomm_reduce_scatter_fail_datatype);
  tcase_add_test(redscf, test_gpucomm_reduce_scatter_fail_optype);
  tcase_add_test(redscf, test_gpucomm_reduce_scatter_fail_src_offset);
  tcase_add_test(redscf, test_gpucomm_reduce_scatter_fail_dest_offset);
  tcase_add_test(redscf, test_gpucomm_reduce_scatter_fail_elemcount);

  TCase* bcasts = tcase_create("test_broadcast");
  tcase_add_unchecked_fixture(bcasts, setup_comm, teardown_comm);
  tcase_add_test(bcasts, test_gpucomm_broadcast_INT);
  tcase_add_test(bcasts, test_gpucomm_broadcast_BYTE);
  tcase_add_test(bcasts, test_gpucomm_broadcast_FLOAT);
  tcase_add_test(bcasts, test_gpucomm_broadcast_DOUBLE);
  tcase_add_test(bcasts, test_gpucomm_broadcast_LONG);
  tcase_add_test(bcasts, test_gpucomm_broadcast_ULONG);

  TCase* bcastf = tcase_create("test_broadcast_fail");
  tcase_add_unchecked_fixture(bcastf, setup_comm, teardown_comm);
  tcase_add_test(bcastf, test_gpucomm_broadcast_fail_datatype);
  tcase_add_test(bcastf, test_gpucomm_broadcast_fail_src_offset);
  tcase_add_test(bcastf, test_gpucomm_broadcast_fail_elemcount);

  TCase* agats = tcase_create("test_all_gather");
  tcase_add_unchecked_fixture(agats, setup_comm, teardown_comm);
  tcase_add_test(agats, test_gpucomm_all_gather_INT);
  tcase_add_test(agats, test_gpucomm_all_gather_BYTE);
  tcase_add_test(agats, test_gpucomm_all_gather_FLOAT);
  tcase_add_test(agats, test_gpucomm_all_gather_DOUBLE);
  tcase_add_test(agats, test_gpucomm_all_gather_LONG);
  tcase_add_test(agats, test_gpucomm_all_gather_ULONG);

  TCase* agatf = tcase_create("test_all_gather_fail");
  tcase_add_unchecked_fixture(agatf, setup_comm, teardown_comm);
  tcase_add_test(agatf, test_gpucomm_all_gather_fail_datatype);
  tcase_add_test(agatf, test_gpucomm_all_gather_fail_src_offset);
  tcase_add_test(agatf, test_gpucomm_all_gather_fail_dest_offset);
  tcase_add_test(agatf, test_gpucomm_all_gather_fail_elemcount);

  suite_add_tcase(s, helps);
  suite_add_tcase(s, reds);
  suite_add_tcase(s, redf);
  suite_add_tcase(s, areds);
  suite_add_tcase(s, aredf);
  suite_add_tcase(s, redscs);
  suite_add_tcase(s, redscf);
  suite_add_tcase(s, bcasts);
  suite_add_tcase(s, bcastf);
  suite_add_tcase(s, agats);
  suite_add_tcase(s, agatf);
  return s;
}
