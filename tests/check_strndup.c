#include <check.h>
#include <util/strndup.h>
#include <stdlib.h>

START_TEST(test_strndup)
{
  const char *str = "test string";
  int str_len = strlen(str);

  // copy str_len characters
  char *copy = strndup(str, str_len);
  int copy_len = strlen(copy);
  ck_assert(copy_len == str_len);
  ck_assert(strncmp(str, copy, min(copy_len, str_len)) == 0);
  free(copy);

  // copy fewer than str_len characters
  static const int len1 = 5;
  copy = strndup(str, len1);
  copy_len = strlen(copy);
  ck_assert(copy_len == len1);
  ck_assert(strncmp(str, copy, min(copy_len, str_len)) == 0);
  free(copy);

  // try to copy more than str_len characters
  static const int len2 = 25;
  copy = strndup(str, len2);
  copy_len = strlen(copy);
  // copy_len == str_len
  ck_assert(copy_len == str_len);
  ck_assert(strncmp(str, copy, min(copy_len, str_len)) == 0);
  free(copy);

  // handle empty string correctly
  copy = strndup("", len1);
  ck_assert(copy == NULL);

  // handle NULL pointer correctly
  copy = strndup(NULL, len1);
  ck_assert(copy == NULL);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("strndup");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_strndup);
  suite_add_tcase(s, tc);
  return s;
}
