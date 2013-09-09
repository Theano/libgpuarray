#include "buf.c"

#include <check.h>

#include <string.h>

#include "private.h"

static size_t SIZES[] = {23, INIT_SIZE-1, INIT_SIZE, INIT_SIZE+1};

static const char TEXT[] = "some text!";

START_TEST(test_buf_alloc)
{
  buf b = BUF_INIT;
  int err;
  size_t target = SIZES[_i];
  ck_assert(b.s == NULL);
  ck_assert(b.i == 0);
  ck_assert(b.a == 0);
  err = buf_alloc(&b, target);
  ck_assert(!err);
  ck_assert(b.s != NULL);
  ck_assert(b.a >= target);
  err = buf_appends(&b, TEXT);
  ck_assert(!err);
  target += b.a;
  err = buf_alloc(&b, target);
  ck_assert(!err);
  ck_assert(b.a >= target);
  ck_assert(strncmp(b.s, TEXT, sizeof(TEXT)-1) == 0);
  buf_free(&b);
}
END_TEST

START_TEST(test_buf_ensurefree)
{
  buf b = BUF_INIT;
  int err;
  size_t target = SIZES[_i];
  err = buf_ensurefree(&b, target);
  ck_assert(!err);
  ck_assert(b.a - b.i >= target);
  buf_free(&b);
}
END_TEST

static const char DATA[] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8};

START_TEST(test_buf_append)
{
  buf b = BUF_INIT;
  buf b2 = BUF_INIT;
  int err, i;

  err = buf_appendb(&b, DATA, sizeof(DATA));
  ck_assert(!err);
  ck_assert(b.a >= sizeof(DATA));
  ck_assert(b.i == sizeof(DATA));
  for (i = 0; i < sizeof(DATA); i++) {
    ck_assert(b.s[i] == DATA[i]);
  }

  err = buf_appends(&b2, TEXT);
  ck_assert(!err);
  ck_assert(strncmp(b2.s, TEXT, sizeof(TEXT)-1) == 0);

  err = buf_append(&b, &b2);
  ck_assert(!err);
  ck_assert(b.a >= sizeof(DATA)+sizeof(TEXT)-1);
  ck_assert(b.i == sizeof(DATA)+sizeof(TEXT)-1);
  for (i = 0; i < sizeof(DATA); i++) {
    ck_assert(b.s[i] == DATA[i]);
  }
  for (i = 0; i < sizeof(TEXT)-1; i++) {
    ck_assert(b.s[i+sizeof(DATA)] == TEXT[i]);
  }

  err = buf_appendc(&b, 'c');
  ck_assert(!err);
  ck_assert(b.a >= sizeof(DATA)+sizeof(TEXT));
  ck_assert(b.i == sizeof(DATA)+sizeof(TEXT));
  for (i = 0; i < sizeof(DATA); i++) {
    ck_assert(b.s[i] == DATA[i]);
  }
  for (i = 0; i < sizeof(TEXT)-1; i++) {
    ck_assert(b.s[i+sizeof(DATA)] == TEXT[i]);
  }
  ck_assert(b.s[sizeof(DATA)+sizeof(TEXT)-1] == 'c');
}
END_TEST

START_TEST(test_buf_free)
{
  buf b = BUF_INIT;
  buf_free(&b);
  ck_assert(b.s == NULL);
  ck_assert(b.a == 0);
  ck_assert(b.i == 0);
  buf_alloc(&b, 23);
  buf_free(&b);
  ck_assert(b.s == NULL);
  ck_assert(b.a == 0);
  ck_assert(b.i == 0);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("buf");
  TCase *tc = tcase_create("All");
  tcase_add_loop_test(tc, test_buf_alloc, 0, nelems(SIZES));
  tcase_add_loop_test(tc, test_buf_ensurefree, 0, nelems(SIZES));
  tcase_add_test(tc, test_buf_append);
  tcase_add_test(tc, test_buf_free);
  suite_add_tcase(s, tc);
  
  return s;
}
