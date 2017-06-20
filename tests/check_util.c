#include <stdlib.h>

#include <check.h>

#include "gpuarray/buffer.h"
#include "gpuarray/util.h"

START_TEST(test_register_type) {
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

START_TEST(test_type_flags) {
  ck_assert_int_eq(gpuarray_type_flags(-1), 0);
  ck_assert_int_eq(gpuarray_type_flags(GA_FLOAT, -1), 0);
  ck_assert_int_eq(gpuarray_type_flags(GA_DOUBLE, -1), GA_USE_DOUBLE);
  ck_assert_int_eq(gpuarray_type_flags(GA_CFLOAT, -1), GA_USE_COMPLEX);
  ck_assert_int_eq(gpuarray_type_flags(GA_CDOUBLE, -1),
                   GA_USE_DOUBLE|GA_USE_COMPLEX);
  ck_assert_int_eq(gpuarray_type_flags(GA_HALF, -1),
                   GA_USE_HALF|GA_USE_SMALL);
  ck_assert_int_eq(gpuarray_type_flags(GA_BYTE, -1), GA_USE_SMALL);
  ck_assert_int_eq(gpuarray_type_flags(GA_SHORT, GA_DOUBLE, -1),
                   GA_USE_SMALL|GA_USE_DOUBLE);
  ck_assert_int_eq(gpuarray_type_flags(GA_DOUBLE, GA_DOUBLE, -1),
                   GA_USE_DOUBLE);
}
END_TEST

START_TEST(test_elemwise_collapse) {
  size_t dims[3];
  ssize_t *strs[2];
  ssize_t _strs0[3];
  ssize_t _strs1[3];
  unsigned int nd;

  strs[0] = _strs0;
  strs[1] = _strs1;

  nd = 3;
  dims[0] = 50;
  dims[1] = 1;
  dims[2] = 20;
  strs[0][0] = 80;
  strs[0][1] = 80;
  strs[0][2] = 4;
  strs[1][0] = 80;
  strs[1][1] = 80;
  strs[1][2] = 4;

  gpuarray_elemwise_collapse(2, &nd, dims, strs);
  ck_assert_uint_eq(nd, 1);
  ck_assert_uint_eq(dims[0], 1000);
  ck_assert_int_eq(strs[0][0], 4);
  ck_assert_int_eq(strs[1][0], 4);

  nd = 3;
  dims[0] = 50;
  dims[1] = 1;
  dims[2] = 20;
  strs[0][0] = 168;
  strs[0][1] = 80;
  strs[0][2] = 4;
  strs[1][0] = 80;
  strs[1][1] = 80;
  strs[1][2] = 4;

  gpuarray_elemwise_collapse(2, &nd, dims, strs);
  ck_assert_uint_eq(nd, 2);
  ck_assert_uint_eq(dims[0], 50);
  ck_assert_uint_eq(dims[1], 20);
  ck_assert_int_eq(strs[0][0], 168);
  ck_assert_int_eq(strs[0][1], 4);
  ck_assert_int_eq(strs[1][0], 80);
  ck_assert_int_eq(strs[1][1], 4);

  nd = 3;
  dims[0] = 20;
  dims[1] = 1;
  dims[2] = 50;
  strs[0][0] = 4;
  strs[0][1] = 80;
  strs[0][2] = 168;
  strs[1][0] = 4;
  strs[1][1] = 80;
  strs[1][2] = 80;

  gpuarray_elemwise_collapse(2, &nd, dims, strs);
  ck_assert_uint_eq(nd, 2);
  ck_assert_uint_eq(dims[0], 20);
  ck_assert_uint_eq(dims[1], 50);
  ck_assert_int_eq(strs[0][0], 4);
  ck_assert_int_eq(strs[0][1], 168);
  ck_assert_int_eq(strs[1][0], 4);
  ck_assert_int_eq(strs[1][1], 80);

  nd = 2;
  dims[0] = 1;
  dims[1] = 1;
  strs[0][0] = 4;
  strs[0][1] = 4;

  gpuarray_elemwise_collapse(1, &nd, dims, strs);
  ck_assert_uint_eq(nd, 1);
  ck_assert_uint_eq(dims[0], 1);
  ck_assert_int_eq(strs[0][0], 4);
}
END_TEST

START_TEST(test_float2half) {
  const float f[] = {
    2.9831426e-08f,
    2e-25f,
    2e-26f,
    1.0005035f,
    1.0002441f,
    65519.f,
    65520.f,
  };
  const ga_half_t h[] = {
    {0x0001u}, /* 2e-24 */
    {0x0000u}, /* 0 */
    {0x0000u}, /* 0 */
    {0x3c01u}, /* 1.0 + 2e-10 */
    {0x3c00u}, /* 1.0 */
    {0x7bffu}, /* 65504 */
    {0x7c00u}, /* Inf */
  };
  unsigned int i;
  ga_half_t hr;

  for (i = 0; i < sizeof(f)/sizeof(f[0]); i++) {
    hr = ga_float2half(f[i]);
    ck_assert_int_eq(hr.h, h[i].h);
  }
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("util");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_register_type);
  tcase_add_test(tc, test_type_flags);
  tcase_add_test(tc, test_elemwise_collapse);
  tcase_add_test(tc, test_float2half);
  suite_add_tcase(s, tc);
  return s;
}
