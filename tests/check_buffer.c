#include <check.h>

#include "compyte/buffer.h"
#include "compyte/error.h"
#include "private.h"

START_TEST(test_get_ops)
{
  const compyte_buffer_ops *ops;
  int valid_ops = 0;
  ops = compyte_get_ops("cuda");
  if (ops != NULL) valid_ops++;
  ops = compyte_get_ops("opencl");
  if (ops != NULL) valid_ops++;
  ck_assert_msg(valid_ops > 0, "No backends are available");

  ops = compyte_get_ops("potato");
  ck_assert(ops == NULL);
}
END_TEST

static char *BACKENDS[] = {"opencl", "cuda"};

START_TEST(test_gpu_error)
{
  const compyte_buffer_ops *ops;
  const char *msg;
  ops = compyte_get_ops(BACKENDS[_i]);
  if (ops == NULL) return;
  msg = Gpu_error(ops, NULL, -1);
  msg = Gpu_error(ops, NULL, 99);
  msg = Gpu_error(ops, NULL, GA_NO_ERROR);
  ck_assert_str_eq(msg, "No error");
}
END_TEST

static const compyte_buffer_ops *ops;
static void *ctx;

static int setup(int i) {
  int j;
  ops = compyte_get_ops(BACKENDS[i]);
  if (ops == NULL)
    return 0;
  for (j = 0; j < 5; j++) {
    ctx = ops->buffer_init(j, NULL);
    if (ctx != NULL)
      return 1;
  }
  return 0;
}

static void teardown(void) {
  if (ctx != NULL) {
    ops->buffer_deinit(ctx);
    ctx = NULL;
  }
}

START_TEST(test_buffer_init)
{
  if (setup(_i)) {
    ops->buffer_deinit(ctx);
    ctx = NULL;
  }
  teardown();
}
END_TEST

START_TEST(test_buffer_alloc)
{
  gpudata *d;

  if (setup(_i)) {
    d = ops->buffer_alloc(ctx, (size_t)-1, NULL);
    ck_assert(d == NULL);

    d = ops->buffer_alloc(ctx, 0, NULL);
    ck_assert(d != NULL);
    ops->buffer_free(d);

    d = ops->buffer_alloc(ctx, 1, NULL);
    ck_assert(d != NULL);
    ops->buffer_free(d);

    d = ops->buffer_alloc(ctx, 1024, NULL);
    ck_assert(d != NULL);
    ops->buffer_free(d);
  }
  teardown();
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("buffer");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_get_ops);
  tcase_add_loop_test(tc, test_gpu_error, 0, nelems(BACKENDS));
  suite_add_tcase(s, tc);
  tc = tcase_create("API");
  tcase_add_loop_test(tc, test_buffer_init, 0, nelems(BACKENDS));
  tcase_add_loop_test(tc, test_buffer_alloc, 0, nelems(BACKENDS));
  suite_add_tcase(s, tc);
  return s;
}
