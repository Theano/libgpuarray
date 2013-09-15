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

static unsigned int refcnt(gpudata *b) {
  unsigned int res;
  int err;
  err = ops->property(NULL, b, NULL, GA_BUFFER_PROP_REFCNT, &res);
  ck_assert(err == GA_NO_ERROR);
  return res;
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
    d = ops->buffer_alloc(ctx, 0, NULL, 0, NULL);
    ck_assert(d != NULL);
    ck_assert_int_eq(refcnt(d), 1);
    ops->buffer_release(d);

    d = ops->buffer_alloc(ctx, 1, NULL, 0, NULL);
    ck_assert(d != NULL);
    ck_assert_int_eq(refcnt(d), 1);
    ops->buffer_release(d);

    d = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d != NULL);
    ck_assert_int_eq(refcnt(d), 1);
    ops->buffer_release(d);
  }
  teardown();
}
END_TEST

START_TEST(test_buffer_retain_release)
{
  gpudata *d;
  gpudata *d2;

  if (setup(_i)) {
    d = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d != NULL);
    ck_assert_int_eq(refcnt(d), 1);

    d2 = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d2 != NULL);
    ck_assert_int_eq(refcnt(d2), 1);

    ops->buffer_retain(d);
    ck_assert_int_eq(refcnt(d), 2);

    ops->buffer_release(d);
    ck_assert_int_eq(refcnt(d), 1);

    ops->buffer_retain(d);
    ops->buffer_retain(d2);
    ops->buffer_retain(d);
    ck_assert_int_eq(refcnt(d), 3);
    ck_assert_int_eq(refcnt(d2), 2);

    ops->buffer_release(d);
    ck_assert_int_eq(refcnt(d), 2);
    ck_assert_int_eq(refcnt(d2), 2);

    ops->buffer_release(d);
    ops->buffer_release(d2);
    ck_assert_int_eq(refcnt(d), 1);
    ck_assert_int_eq(refcnt(d2), 1);

    ops->buffer_release(d);
    ck_assert_int_eq(refcnt(d2), 1);

    ops->buffer_release(d2);
  }
  teardown();
}
END_TEST

START_TEST(test_buffer_share)
{
  gpudata *d;
  gpudata *d2;

  if (setup(_i)) {
    d = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d != NULL);
    d2 = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d2 != NULL);

    ck_assert_int_eq(ops->buffer_share(d, d2, NULL), 0);
    ck_assert_int_eq(ops->buffer_share(d, d, NULL), 1);

    /* TODO: test for OpenCL subbuffers and Cuda overlapping allocations */
  }
  teardown();
}
END_TEST

START_TEST(test_buffer_read_write)
{
  const int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int32_t buf[nelems(data)];
  gpudata *d;
  int err;
  unsigned int i;

  if (setup(_i)) {
    d = ops->buffer_alloc(ctx, sizeof(data), NULL, 0, NULL);
    ck_assert(d != NULL);

    err = ops->buffer_write(d, 0, data, sizeof(data));
    ck_assert_int_eq(err, GA_NO_ERROR);

    memset(buf, 0, sizeof(data));
    err = ops->buffer_read(buf, d, 0, sizeof(data));
    ck_assert_int_eq(err, GA_NO_ERROR);
    for (i = 0; i < nelems(data); i++) {
      ck_assert_int_eq(data[i], buf[i]);
    }

    memset(buf, 0, sizeof(data));
    err = ops->buffer_read(buf, d, sizeof(int32_t), sizeof(data)-sizeof(int32_t));
    ck_assert_int_eq(err, GA_NO_ERROR);
    for (i = 0; i < nelems(data)-1; i++) {
      ck_assert_int_eq(data[i+1], buf[i]);

    }

    err = ops->buffer_write(d, sizeof(int32_t)*2, data, sizeof(data)-(sizeof(int32_t)*2));
    ck_assert_int_eq(err, GA_NO_ERROR);

    memset(buf, 0, sizeof(data));
    err = ops->buffer_read(buf, d, 0, sizeof(data));
    ck_assert_int_eq(err, GA_NO_ERROR);
    for (i = 0; i < nelems(data)-2; i++) {
      ck_assert_int_eq(data[i], buf[i+2]);
    }
    for (i = 0; i < 2; i++) {
      ck_assert_int_eq(data[i], buf[i]);
    }
    ops->buffer_release(d);
  }
  teardown();
}
END_TEST

START_TEST(test_buffer_move)
{
  const int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int32_t buf[nelems(data)];
  gpudata *d;
  gpudata *d2;
  int err;
  unsigned int i;

  if (setup(_i)) {
    d = ops->buffer_alloc(ctx, sizeof(data), NULL, 0, NULL);
    ck_assert(d != NULL);
    d2 = ops->buffer_alloc(ctx, sizeof(data)*2, NULL, 0, NULL);
    ck_assert(d != NULL);

    err = ops->buffer_write(d, 0, data, sizeof(data));
    ck_assert(err == GA_NO_ERROR);

    err = ops->buffer_move(d2, sizeof(data), d, 0, sizeof(data));
    ck_assert(err == GA_NO_ERROR);

    err = ops->buffer_read(buf, d2, sizeof(data), sizeof(data));
    ck_assert(err == GA_NO_ERROR);
    for (i = 0; i < nelems(data); i++) {
      ck_assert_int_eq(buf[i], data[i]);
    }

    err = ops->buffer_move(d2, 0, d, sizeof(uint32_t), sizeof(data)-sizeof(uint32_t));
    ck_assert(err == GA_NO_ERROR);

    err = ops->buffer_read(buf, d2, 0, sizeof(data));
    ck_assert(err == GA_NO_ERROR);
    for (i = 0; i < nelems(data)-1; i++) {
      ck_assert_int_eq(buf[i], data[i+1]);
    }

    ops->buffer_release(d);
    ops->buffer_release(d2);
  }
  teardown();
}
END_TEST

static const char *KERNEL = "KERNEL void k(GLOBAL_MEM ga_float *a, ga_float b, GLOBAL_MEM ga_float *c) {}";

START_TEST(test_kernel_setargs)
{
  gpukernel *k;
  gpudata *d;
  gpudata *d2;
  int err;

  if (setup(_i)) {
    k = ops->kernel_alloc(ctx, 1, &KERNEL, NULL, "k", GA_USE_CLUDA, &err);
    printf("%s\n", Gpu_error(ops, ctx, err));
    ck_assert(k != NULL);
    d = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d != NULL);
    d2 = ops->buffer_alloc(ctx, 1024, NULL, 0, NULL);
    ck_assert(d != NULL);

    ck_assert_int_eq(refcnt(d), 1);
    ck_assert_int_eq(refcnt(d2), 1);

    err = ops->kernel_setarg(k, 0, GA_BUFFER, d);
    ck_assert_int_eq(GA_NO_ERROR, err);
    ck_assert_int_eq(refcnt(d), 2);

    err = ops->kernel_setarg(k, 0, GA_BUFFER, d);
    ck_assert_int_eq(GA_NO_ERROR, err);
    ck_assert_int_eq(refcnt(d), 2);

    err = ops->kernel_setarg(k, 2, GA_BUFFER, d);
    ck_assert_int_eq(GA_NO_ERROR, err);
    ck_assert_int_eq(refcnt(d), 3);

    err = ops->kernel_setarg(k, 2, GA_BUFFER, d2);
    ck_assert_int_eq(GA_NO_ERROR, err);
    ck_assert_int_eq(refcnt(d), 2);
    ck_assert_int_eq(refcnt(d2), 2);

    ops->kernel_release(k);
    ck_assert_int_eq(refcnt(d), 1);
    ck_assert_int_eq(refcnt(d2), 1);

    ops->buffer_release(d);
    ops->buffer_release(d2);
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
  tcase_add_loop_test(tc, test_buffer_retain_release, 0, nelems(BACKENDS));
  tcase_add_loop_test(tc, test_buffer_share, 0, nelems(BACKENDS));
  tcase_add_loop_test(tc, test_buffer_read_write, 0, nelems(BACKENDS));
  tcase_add_loop_test(tc, test_buffer_move, 0, nelems(BACKENDS));
  tcase_add_loop_test(tc, test_kernel_setargs, 0, nelems(BACKENDS));
  suite_add_tcase(s, tc);
  return s;
}
