#include <check.h>

#include "gpuarray/array.h"
#include "gpuarray/buffer.h"
#include "gpuarray/elemwise.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

extern void *ctx;

void setup(void);
void teardown(void);

#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)
/* float 16 table (0 through 10) */
static const uint16_t F16[10] = {0x0000, 0x3c00, 0x4000, 0x4200, 0x4400,
                                 0x4500, 0x4600, 0x4700, 0x4800, 0x4880};

START_TEST(test_contig_simple) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[3] = {1, 2, 3};
  static const uint32_t data2[3] = {4, 5, 6};
  uint32_t data3[3] = {0};

  size_t dims[1];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 1, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 5);
  ck_assert_int_eq(data3[1], 7);
  ck_assert_int_eq(data3[2], 9);
}
END_TEST

START_TEST(test_contig_f16) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static uint16_t data1[3];
  data1[0] = F16[1];
  data1[1] = F16[2];
  data1[2] = F16[3];
  static uint16_t data2[3];
  data2[0] = F16[4];
  data2[1] = F16[5];
  data2[2] = F16[6];
  uint16_t data3[3] = {0};

  size_t dims[1];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_HALF, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_HALF, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_HALF, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_HALF;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_HALF;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_HALF;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 1, GE_CONVERT_F16);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], F16[5]);
  ck_assert_int_eq(data3[1], F16[7]);
  ck_assert_int_eq(data3[2], F16[9]);
}
END_TEST

START_TEST(test_contig_0) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  size_t dims[1];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 0;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 1, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));
}
END_TEST

START_TEST(test_basic_simple) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[3] = {1, 2, 3};
  static const uint32_t data2[3] = {4, 5, 6};
  uint32_t data3[3] = {0};

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 1;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 2, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 5);
  ck_assert_int_eq(data3[1], 7);
  ck_assert_int_eq(data3[2], 9);
}
END_TEST

START_TEST(test_basic_f16) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static uint16_t data1[3];
  data1[0] = F16[1];
  data1[1] = F16[2];
  data1[2] = F16[3];
  static uint16_t data2[3];
  data2[0] = F16[4];
  data2[1] = F16[5];
  data2[2] = F16[6];
  uint16_t data3[3] = {0};

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 1;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_HALF, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_HALF, 2, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_HALF, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_HALF;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_HALF;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_HALF;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, GE_CONVERT_F16);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], F16[5]);
  ck_assert_int_eq(data3[1], F16[7]);
  ck_assert_int_eq(data3[2], F16[9]);
}
END_TEST

START_TEST(test_basic_offset) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[3] = {1, 2, 3};
  static const uint32_t data2[3] = {4, 5, 6};
  uint32_t data3[3] = {0};

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 1;
  dims[1] = 6;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  /* Simulate indexing */
  a.offset = 12;
  a.dimensions[1] = 3;

  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 2, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 5);
  ck_assert_int_eq(data3[1], 7);
  ck_assert_int_eq(data3[2], 9);
}
END_TEST

START_TEST(test_basic_remove1) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[6] = {1, 2, 3, 4, 5, 6};
  static const uint32_t data2[6] = {7, 8, 9, 10, 11, 12};
  uint32_t data3[6] = {0};

  size_t dims[4];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 1;
  dims[1] = 3;
  dims[2] = 2;
  dims[3] = 1;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 4, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 4, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 4, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 0, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, 0));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 8);
  ck_assert_int_eq(data3[1], 12);
  ck_assert_int_eq(data3[2], 11);
  ck_assert_int_eq(data3[3], 15);
  ck_assert_int_eq(data3[4], 14);
  ck_assert_int_eq(data3[5], 18);
}
END_TEST

START_TEST(test_basic_broadcast) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[3] = {1, 2, 3};
  static const uint32_t data2[2] = {4, 5};
  uint32_t data3[6] = {0};

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 1;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  dims[0] = 2;
  dims[1] = 1;

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 2, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  dims[0] = 2;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ck_assert_int_eq(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE), GA_VALUE_ERROR);

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_NOCOLLAPSE | GE_BROADCAST));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 5);
  ck_assert_int_eq(data3[1], 6);
  ck_assert_int_eq(data3[2], 7);
  ck_assert_int_eq(data3[3], 6);
  ck_assert_int_eq(data3[4], 7);
  ck_assert_int_eq(data3[5], 8);
}
END_TEST

START_TEST(test_basic_collapse) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[6] = {1, 2, 3, 4, 5, 6};
  static const uint32_t data2[6] = {7, 8, 9, 10, 11, 12};
  uint32_t data3[6] = {0};

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 2;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, 0));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 8);
  ck_assert_int_eq(data3[1], 10);
  ck_assert_int_eq(data3[2], 12);
  ck_assert_int_eq(data3[3], 14);
  ck_assert_int_eq(data3[4], 16);
  ck_assert_int_eq(data3[5], 18);
}
END_TEST

START_TEST(test_basic_neg_strides) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  static const uint32_t data1[6] = {1, 2, 3, 4, 5, 6};
  static const uint32_t data2[6] = {7, 8, 9, 10, 11, 12};
  uint32_t data3[6] = {0};

  size_t dims[1];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  ssize_t starts[1];
  ssize_t stops[1];
  ssize_t steps[1];

  dims[0] = 6;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  starts[0] = 5;
  stops[0] = -1;
  steps[0] = -1;

  ga_assert_ok(GpuArray_index_inplace(&b, starts, stops, steps));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 1, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, 0));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 13);
  ck_assert_int_eq(data3[1], 13);
  ck_assert_int_eq(data3[2], 13);
  ck_assert_int_eq(data3[3], 13);
  ck_assert_int_eq(data3[4], 13);
  ck_assert_int_eq(data3[5], 13);
}
END_TEST

START_TEST(test_basic_0) {
  GpuArray a;
  GpuArray b;
  GpuArray c;

  GpuElemwise *ge;

  size_t dims[2];

  gpuelemwise_arg args[3] = {{0}};
  void *rargs[3];

  dims[0] = 0;
  dims[1] = 3;

  ga_assert_ok(GpuArray_empty(&a, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_empty(&b, ctx, GA_UINT, 2, dims, GA_F_ORDER));

  ga_assert_ok(GpuArray_empty(&c, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ctx, "", "c = a + b", 3, args, 2, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, 0));
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("elemwise");
  TCase *tc = tcase_create("contig");
  tcase_set_timeout(tc, 8.0);
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_contig_simple);
  tcase_add_test(tc, test_contig_f16);
  tcase_add_test(tc, test_contig_0);
  suite_add_tcase(s, tc);
  tc = tcase_create("basic");
  tcase_set_timeout(tc, 8.0);
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_basic_simple);
  tcase_add_test(tc, test_basic_f16);
  tcase_add_test(tc, test_basic_offset);
  tcase_add_test(tc, test_basic_remove1);
  tcase_add_test(tc, test_basic_broadcast);
  tcase_add_test(tc, test_basic_collapse);
  tcase_add_test(tc, test_basic_neg_strides);
  tcase_add_test(tc, test_basic_0);
  suite_add_tcase(s, tc);
  return s;
}
