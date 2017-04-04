#include <stdlib.h>

#include <check.h>

#include "gpuarray/array.h"
#include "gpuarray/blas.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

extern void *ctx;

void setup(void);
void teardown(void);

#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)

static inline void ck_assert_fbuf_eq(const float *b, const float *r,
                                     unsigned int n) {
  unsigned int i;
  for (i = 0; i < n; i++) {
    ck_assert_msg(b[i] == r[i], "Difference at %u: %f != %f(ref)", i, b[i], r[i]);
  }
}

START_TEST(test_gemmBatch_3d_C) {
  GpuArray A;
  GpuArray B;
  GpuArray C;

  size_t dims[3] = {2, 3, 3};
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float res[] = {30, 36, 42, 66, 81, 96, 102, 126, 150,
                       30, 36, 42, 66, 81, 96, 102, 126, 150};

  ga_assert_ok(GpuArray_empty(&A, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_empty(&B, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_empty(&C, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_write(&A, data, sizeof(data)));
  ga_assert_ok(GpuArray_write(&B, data, sizeof(data)));

  ga_assert_ok(GpuArray_rgemmBatch_3d(cb_no_trans, cb_no_trans, 1, &A, &B, 0, &C, 1));

  ga_assert_ok(GpuArray_read(data, sizeof(data), &C));

  ck_assert_fbuf_eq(data, res, sizeof(res)/sizeof(float));
}
END_TEST

START_TEST(test_gemmBatch_3d_F) {
  GpuArray A;
  GpuArray B;
  GpuArray C;

  size_t dims[3] = {2, 3, 3};
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float res[] = {42, 78, 78, 60, 114, 114, 51, 69, 96,
                       66, 39, 111, 54, 54, 90, 78, 78, 132};

  ga_assert_ok(GpuArray_empty(&A, ctx, GA_FLOAT, 3, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_empty(&B, ctx, GA_FLOAT, 3, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_empty(&C, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_write(&A, data, sizeof(data)));
  ga_assert_ok(GpuArray_write(&B, data, sizeof(data)));

  ga_assert_ok(GpuArray_rgemmBatch_3d(cb_no_trans, cb_no_trans, 1, &A, &B, 0, &C, 0));

  ga_assert_ok(GpuArray_read(data, sizeof(data), &C));

  ck_assert_fbuf_eq(data, res, sizeof(res)/sizeof(float));
}
END_TEST

START_TEST(test_gemmBatch_3d_S) {
  GpuArray A;
  GpuArray B;
  GpuArray C;
  ssize_t t;

  size_t dims[3] = {2, 3, 3};
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float res[] = {14, 32, 50, 50, 122, 194, 32, 77, 122,
                       26, 62, 98, 17, 53, 89, 44, 107, 170};

  ga_assert_ok(GpuArray_empty(&A, ctx, GA_FLOAT, 3, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_empty(&B, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_empty(&C, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_write(&A, data, sizeof(data)));
  ga_assert_ok(GpuArray_write(&B, data, sizeof(data)));

  A.strides[0] = 8;
  A.strides[1] = 24;
  A.strides[2] = 4;
  GpuArray_fix_flags(&A);

  t = B.strides[1];
  B.strides[1] = B.strides[2];
  B.strides[2] = t;
  GpuArray_fix_flags(&B);

  ga_assert_ok(GpuArray_rgemmBatch_3d(cb_no_trans, cb_no_trans, 1, &A, &B, 0, &C, 1));

  ga_assert_ok(GpuArray_read(data, sizeof(data), &C));

  ck_assert_fbuf_eq(data, res, sizeof(res)/sizeof(float));
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("blas");
  TCase *tc = tcase_create("all");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_set_timeout(tc, 16.0);
  tcase_add_test(tc, test_gemmBatch_3d_C);
  tcase_add_test(tc, test_gemmBatch_3d_F);
  tcase_add_test(tc, test_gemmBatch_3d_S);
  suite_add_tcase(s, tc);
  return s;
}
