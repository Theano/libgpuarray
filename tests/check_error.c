#include <check.h>

#include "gpuarray/error.h"

START_TEST(test_error_str) {
  const char *msg;

  msg = gpuarray_error_str(-1);
  ck_assert_str_eq(msg, "Unknown GA error");
  msg = gpuarray_error_str(99);
  ck_assert_str_eq(msg, "Unknown GA error");
  msg = gpuarray_error_str(GA_NO_ERROR);
  ck_assert_str_eq(msg, "No error");
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("error");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_error_str);
  suite_add_tcase(s, tc);
  return s;
}
