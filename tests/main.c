#include <stdlib.h>
#ifdef TEST_COLLECTIVES
#include <stdio.h>
#endif  // TEST_COLLECTIVES

#include <check.h>
#ifdef TEST_COLLECTIVES
#include <mpi.h>

extern int comm_ndev;
extern int comm_rank;
extern char *dev_name;
#endif  // TEST_COLLECTIVES
extern Suite *get_suite(void);

int main(int argc, char *argv[])
{
#ifdef TEST_COLLECTIVES
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_ndev);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  if (argc < comm_ndev) {
    if (comm_rank == 0)
      printf("Usage : %s <GPU list per rank>\n", argv[0]);
    exit(1);
  }

  dev_name = argv[comm_rank + 1];  // Set a gpu for this process.
#endif  // TEST_COLLECTIVES

  int number_failed;
  Suite *s = get_suite();
  SRunner *sr = srunner_create(s);
#ifdef TEST_COLLECTIVES
  // Check by default forks to another (non mpi registered) process in order to
  // run tests. Using MPI inside tests means we must disable this.
  srunner_set_fork_status(sr, CK_NOFORK);
#endif  // TEST_COLLECTIVES
  srunner_run_all(sr, CK_VERBOSE);
  number_failed = srunner_ntests_failed(sr);
  srunner_free(sr);

#ifdef TEST_COLLECTIVES
  MPI_Finalize();
#endif  // TEST_COLLECTIVES
  return number_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
