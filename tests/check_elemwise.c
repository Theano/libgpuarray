#include <check.h>

#include <gpuarray/buffer.h>
#include <gpuarray/array.h>
#include <gpuarray/elemwise.h>
#include <gpuarray/error.h>
#include <gpuarray/types.h>

extern void *ctx;
extern const gpuarray_buffer_ops *ops;

void setup(void);
void teardown(void);

#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)


START_TEST(test_contig_simple)
{
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

  ga_assert_ok(GpuArray_empty(&a, ops, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ops, ctx, GA_UINT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ops, ctx, GA_UINT, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].nd = 1;
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].nd = 1;
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].nd = 1;
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ops, ctx, "", "c = a + b", 3, args, 0);

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


START_TEST(test_basic_simple)
{
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

  ga_assert_ok(GpuArray_empty(&a, ops, ctx, GA_UINT, 2, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ops, ctx, GA_UINT, 2, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ops, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].nd = 2;
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].nd = 2;
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].nd = 2;
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ops, ctx, "", "c = a + b", 3, args, 0);

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


START_TEST(test_basic_remove1)
{
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

  ga_assert_ok(GpuArray_empty(&a, ops, ctx, GA_UINT, 4, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ops, ctx, GA_UINT, 4, dims, GA_F_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ops, ctx, GA_UINT, 4, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].nd = 4;
  args[0].typecode = GA_UINT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].nd = 4;
  args[1].typecode = GA_UINT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].nd = 4;
  args[2].typecode = GA_UINT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ops, ctx, "", "c = a + b", 3, args, 0);

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

Suite *get_suite(void) {
  Suite *s = suite_create("elemwise");
  TCase *tc = tcase_create("contig");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_contig_simple);
  suite_add_tcase(s, tc);
  tc = tcase_create("basic");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_basic_simple);
  tcase_add_test(tc, test_basic_remove1);
  suite_add_tcase(s, tc);
  return s;
}


