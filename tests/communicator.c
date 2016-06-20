#include <check.h>
#include <mpi.h>

#include "gpuarray/buffer.h"
#include "gpuarray/buffer_collectives.h"
#include "gpuarray/error.h"

extern gpucontext* ctx;
int comm_ndev;  //!< number of devices in the comm
int comm_rank;  //!< comm's rank in the world
// (for the tests it's the same as process rank in MPI_COMM_WORLD)
gpucomm* comm;

extern void setup(void);
extern void teardown(void);

/**
 * \brief Setup for `check_buffer_collectives.c` and `check_collectives.c`.
 *
 * Includes tests for `gpucomm_new` and `gpucomm_gen_clique_id`
 */
void setup_comm(void)
{
  setup();
  int err;

  MPI_Barrier(MPI_COMM_WORLD);
  gpucommCliqueId comm_id;
  err = gpucomm_gen_clique_id(ctx, &comm_id);
  // Has successfully got a unique comm id.
  ck_assert_int_eq(err, GA_NO_ERROR);

  MPI_Bcast(&comm_id, GA_COMM_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);
  err = gpucomm_new(&comm, ctx, comm_id, comm_ndev, comm_rank % comm_ndev);
  // Has successfully created a new gpucomm.
  ck_assert_int_eq(err, GA_NO_ERROR);
  ck_assert_ptr_ne(comm, NULL);
}

void teardown_comm(void)
{
  gpucomm_free(comm);
  teardown();
}
