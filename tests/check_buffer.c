#include <check.h>

#include "compyte/buffer.h"
#include "compyte/error.h"
#include "private.h"

START_TEST(test_get_ops)
{
  const compyte_buffer_ops *ops;
  /* We can't assert anything since we don't know if the ops are built
     or not */
  ops = compyte_get_ops("cuda");
  ops = compyte_get_ops("opencl");
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

Suite *get_suite(void) {
  Suite *s = suite_create("buffer");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_get_ops);
  tcase_add_loop_test(tc, test_gpu_error, 0, nelems(BACKENDS));
  suite_add_tcase(s, tc);
  return s;
}
