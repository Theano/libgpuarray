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

START_TEST(test_gemmBatch_3d) {
  GpuArray A;
  GpuArray B;
  GpuArray C;

  size_t dims[3] = {32, 32, 32};

  ga_assert_ok(GpuArray_empty(&A, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_empty(&B, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_empty(&C, ctx, GA_FLOAT, 3, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_rgemmBatch_3d(cb_no_trans, cb_no_trans, 1, &A, &B, 0, &C, 1));
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("blas");
  TCase *tc = tcase_create("all");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_set_timeout(tc, 16.0);
  tcase_add_test(tc, test_gemmBatch_3d);
  suite_add_tcase(s, tc);
  return s;
}
