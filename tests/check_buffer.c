#include <check.h>

#include "gpuarray/buffer.h"
#include "gpuarray/error.h"

#include "private.h"

extern void *ctx;

void setup(void);
void teardown(void);

START_TEST(test_gpu_error) {
  const char *msg;
  msg = gpucontext_error(NULL, -1);
  msg = gpucontext_error(NULL, 99);
  msg = gpucontext_error(NULL, GA_NO_ERROR);
  ck_assert_str_eq(msg, "No error");
}
END_TEST

static unsigned int refcnt(gpudata *b) {
  unsigned int res;
  int err;
  err = gpudata_property(b, GA_BUFFER_PROP_REFCNT, &res);
  ck_assert(err == GA_NO_ERROR);
  return res;
}

START_TEST(test_buffer_alloc) {
  gpudata *d;

  d = gpudata_alloc(ctx, 0, NULL, 0, NULL);
  ck_assert(d != NULL);
  ck_assert_int_eq(refcnt(d), 1);
  gpudata_release(d);

  d = gpudata_alloc(ctx, 1, NULL, 0, NULL);
  ck_assert(d != NULL);
  ck_assert_int_eq(refcnt(d), 1);
  gpudata_release(d);

  d = gpudata_alloc(ctx, 1024, NULL, 0, NULL);
  ck_assert(d != NULL);
  ck_assert_int_eq(refcnt(d), 1);
  gpudata_release(d);
}
END_TEST

START_TEST(test_buffer_retain_release) {
  gpudata *d;
  gpudata *d2;

  d = gpudata_alloc(ctx, 1024, NULL, 0, NULL);
  ck_assert(d != NULL);
  ck_assert_int_eq(refcnt(d), 1);

  d2 = gpudata_alloc(ctx, 1024, NULL, 0, NULL);
  ck_assert(d2 != NULL);
  ck_assert_int_eq(refcnt(d2), 1);

  gpudata_retain(d);
  ck_assert_int_eq(refcnt(d), 2);

  gpudata_release(d);
  ck_assert_int_eq(refcnt(d), 1);

  gpudata_retain(d);
  gpudata_retain(d2);
  gpudata_retain(d);
  ck_assert_int_eq(refcnt(d), 3);
  ck_assert_int_eq(refcnt(d2), 2);

  gpudata_release(d);
  ck_assert_int_eq(refcnt(d), 2);
  ck_assert_int_eq(refcnt(d2), 2);

  gpudata_release(d);
  gpudata_release(d2);
  ck_assert_int_eq(refcnt(d), 1);
  ck_assert_int_eq(refcnt(d2), 1);

  gpudata_release(d);
  ck_assert_int_eq(refcnt(d2), 1);

  gpudata_release(d2);
}
END_TEST

START_TEST(test_buffer_share) {
  gpudata *d;
  gpudata *d2;

  d = gpudata_alloc(ctx, 1024, NULL, 0, NULL);
  ck_assert(d != NULL);
  d2 = gpudata_alloc(ctx, 1024, NULL, 0, NULL);
  ck_assert(d2 != NULL);

  ck_assert_int_eq(gpudata_share(d, d2, NULL), 0);
  ck_assert_int_eq(gpudata_share(d, d, NULL), 1);
}
END_TEST

START_TEST(test_buffer_read_write) {
  const int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int32_t buf[nelems(data)];
  gpudata *d;
  int err;
  unsigned int i;

  d = gpudata_alloc(ctx, sizeof(data), NULL, 0, NULL);
  ck_assert(d != NULL);

  err = gpudata_write(d, 0, data, sizeof(data));
  ck_assert_int_eq(err, GA_NO_ERROR);

  memset(buf, 0, sizeof(data));
  err = gpudata_read(buf, d, 0, sizeof(data));
  ck_assert_int_eq(err, GA_NO_ERROR);
  for (i = 0; i < nelems(data); i++) {
    ck_assert_int_eq(data[i], buf[i]);
  }

  memset(buf, 0, sizeof(data));
  err = gpudata_read(buf, d, sizeof(int32_t), sizeof(data) - sizeof(int32_t));
  ck_assert_int_eq(err, GA_NO_ERROR);
  for (i = 0; i < nelems(data) - 1; i++) {
    ck_assert_int_eq(data[i + 1], buf[i]);
  }

  err = gpudata_write(d, sizeof(int32_t) * 2, data,
                      sizeof(data) - (sizeof(int32_t) * 2));
  ck_assert_int_eq(err, GA_NO_ERROR);

  memset(buf, 0, sizeof(data));
  err = gpudata_read(buf, d, 0, sizeof(data));
  ck_assert_int_eq(err, GA_NO_ERROR);
  for (i = 0; i < nelems(data) - 2; i++) {
    ck_assert_int_eq(data[i], buf[i + 2]);
  }
  for (i = 0; i < 2; i++) {
    ck_assert_int_eq(data[i], buf[i]);
  }
  gpudata_release(d);
}
END_TEST

START_TEST(test_buffer_move) {
  const int32_t data[] = {0, 1, 2, 3, 4, 5, 6, 7};
  int32_t buf[nelems(data)];
  gpudata *d;
  gpudata *d2;
  int err;
  unsigned int i;

  d = gpudata_alloc(ctx, sizeof(data), NULL, 0, NULL);
  ck_assert(d != NULL);
  d2 = gpudata_alloc(ctx, sizeof(data) * 2, NULL, 0, NULL);
  ck_assert(d2 != NULL);

  err = gpudata_write(d, 0, data, sizeof(data));
  ck_assert(err == GA_NO_ERROR);

  err = gpudata_move(d2, sizeof(data), d, 0, sizeof(data));
  ck_assert(err == GA_NO_ERROR);

  err = gpudata_read(buf, d2, sizeof(data), sizeof(data));
  ck_assert(err == GA_NO_ERROR);
  for (i = 0; i < nelems(data); i++) {
    ck_assert_int_eq(buf[i], data[i]);
  }

  err =
      gpudata_move(d2, 0, d, sizeof(uint32_t), sizeof(data) - sizeof(uint32_t));
  ck_assert(err == GA_NO_ERROR);

  err = gpudata_read(buf, d2, 0, sizeof(data));
  ck_assert(err == GA_NO_ERROR);
  for (i = 0; i < nelems(data) - 1; i++) {
    ck_assert_int_eq(buf[i], data[i + 1]);
  }

  gpudata_release(d);
  gpudata_release(d2);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("buffer");
  TCase *tc = tcase_create("API");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_add_test(tc, test_gpu_error);
  tcase_add_test(tc, test_buffer_alloc);
  tcase_add_test(tc, test_buffer_retain_release);
  tcase_add_test(tc, test_buffer_share);
  tcase_add_test(tc, test_buffer_read_write);
  tcase_add_test(tc, test_buffer_move);
  suite_add_tcase(s, tc);
  return s;
}
