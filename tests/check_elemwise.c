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

  ga_assert_ok(GpuArray_empty(&a, ops, ctx, GA_FLOAT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&a, data1, sizeof(data1)));

  ga_assert_ok(GpuArray_empty(&b, ops, ctx, GA_FLOAT, 1, dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&b, data2, sizeof(data2)));

  ga_assert_ok(GpuArray_empty(&c, ops, ctx, GA_FLOAT, 1, dims, GA_C_ORDER));

  args[0].name = "a";
  args[0].nd = 1;
  args[0].typecode = GA_FLOAT;
  args[0].flags = GE_READ;

  args[1].name = "b";
  args[1].nd = 1;
  args[1].typecode = GA_FLOAT;
  args[1].flags = GE_READ;

  args[2].name = "c";
  args[2].nd = 1;
  args[2].typecode = GA_FLOAT;
  args[2].flags = GE_WRITE;

  ge = GpuElemwise_new(ops, ctx, "", "c = a + b", 3, args, 0);

  ck_assert_ptr_ne(ge, NULL);

  rargs[0] = &a;
  rargs[1] = &b;
  rargs[2] = &c;

  ga_assert_ok(GpuElemwise_call(ge, rargs, GE_BROADCAST));

  ga_assert_ok(GpuArray_read(data3, sizeof(data3), &c));

  ck_assert_int_eq(data3[0], 5);
  ck_assert_int_eq(data3[1], 7);
  ck_assert_int_eq(data3[2], 9);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("elemwise");
  TCase *tc = tcase_create("contig");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_contig_simple);
  suite_add_tcase(s, tc);
  tc = tcase_create("basic");
  suite_add_tcase(s, tc);
  return s;
}


