#include <check.h>
#include <gpuarray/util.h>
#include <gpuarray/buffer.h>
#include <stdlib.h>

START_TEST(test_register_type)
{
  int tcode;
  gpuarray_type *t = malloc(sizeof(*t));
  ck_assert(t != NULL);
  t->cluda_name = "ga_test";
  t->size = 5;
  t->align = 1;
  t->typecode = 1; /* Normally you don't fill this */
  tcode = gpuarray_register_type(t, NULL);
  ck_assert(tcode != -1);
  ck_assert(tcode == t->typecode);
  ck_assert(gpuarray_get_type(tcode) != NULL);
  ck_assert_str_eq(gpuarray_get_type(tcode)->cluda_name, "ga_test");
}
END_TEST

START_TEST(test_type_flags)
{
  ck_assert_int_eq(gpuarray_type_flags(-1), 0);
  ck_assert_int_eq(gpuarray_type_flags(GA_FLOAT, -1), 0);
  ck_assert_int_eq(gpuarray_type_flags(GA_DOUBLE, -1), GA_USE_DOUBLE);
  ck_assert_int_eq(gpuarray_type_flags(GA_CFLOAT, -1), GA_USE_COMPLEX);
  ck_assert_int_eq(gpuarray_type_flags(GA_CDOUBLE, -1),
                   GA_USE_DOUBLE|GA_USE_COMPLEX);
  ck_assert_int_eq(gpuarray_type_flags(GA_HALF, -1), GA_USE_HALF|GA_USE_SMALL);
  ck_assert_int_eq(gpuarray_type_flags(GA_BYTE, -1), GA_USE_SMALL);
  ck_assert_int_eq(gpuarray_type_flags(GA_SHORT, GA_DOUBLE, -1),
                   GA_USE_SMALL|GA_USE_DOUBLE);
  ck_assert_int_eq(gpuarray_type_flags(GA_DOUBLE, GA_DOUBLE, -1),
                   GA_USE_DOUBLE);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("util");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_register_type);
  tcase_add_test(tc, test_type_flags);
  suite_add_tcase(s, tc);
  return s;
}
