#include <check.h>
#include <stdio.h>

#include "compyte/types.h"
#include "compyte/util.h"
#include "compyte/error.h"

static compyte_type t;
static compyte_type t2;

START_TEST(test_register_type)
{
  int typecode;
  const compyte_type *pt, *pt2;

  /* Check that registration works */
  t.cluda_name = "void";
  t.size = 0xf0f0;
  t.align = 0xabcd;
  typecode = compyte_register_type(&t, NULL);
  ck_assert(typecode != -1);
  ck_assert(t.typecode == typecode);
  pt = compyte_get_type(typecode);
  ck_assert(pt != NULL);
  ck_assert(pt == &t);

  /* Check that a second type does not overwrite the first */
  t2.cluda_name = "potato";
  t2.size = 0x0f0f;
  t2.align = 0xdcba;
  typecode = compyte_register_type(&t2, NULL);
  ck_assert(typecode != -1);
  ck_assert(t2.typecode == typecode);
  ck_assert(t.typecode != typecode);

  /* Check that the first type did not move */
  pt2 = compyte_get_type(t.typecode);
  ck_assert(pt2 == pt);
}
END_TEST

START_TEST(test_get_type)
{
  const compyte_type *pt;
  int typecode;

  pt = compyte_get_type(0);
  ck_assert(pt->typecode == 0);

  pt = compyte_get_type(GA_FLOAT);
  ck_assert(pt->typecode == GA_FLOAT);

  pt = compyte_get_type(GA_NBASE);
  ck_assert(pt->typecode == -1);

  pt = compyte_get_type(GA_DELIM);
  ck_assert(pt->typecode == -1);

  pt = compyte_get_type(GA_DOUBLE2);
  ck_assert(pt->typecode == GA_DOUBLE2);

  pt = compyte_get_type(GA_ENDVEC);
  ck_assert(pt->typecode == -1);

  pt = compyte_get_type(512);
  ck_assert(pt->typecode == -1);

  pt = compyte_get_type(513);
  ck_assert(pt->typecode == -1);
}
END_TEST

START_TEST(test_get_elsize)
{
  ck_assert(compyte_get_elsize(GA_INT) == 4);
  ck_assert(compyte_get_elsize(GA_DELIM) == (size_t)-1);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("types");
  TCase *tc = tcase_create("All");
  tcase_add_test(tc, test_register_type);
  tcase_add_test(tc, test_get_type);
  suite_add_tcase(s, tc);
  return s;
}
