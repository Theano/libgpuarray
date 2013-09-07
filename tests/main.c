#include <check.h>

extern Suite *get_suite(void);

int main(void) {
  int number_failed;
  Suite *s = get_suite();
  SRunner *sr = srunner_create(s);
  srunner_run_all(sr, CK_VERBOSE);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);
  return number_failed;
}
