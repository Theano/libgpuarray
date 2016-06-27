#include <limits.h>
#include <stdlib.h>

#include <check.h>

#include "gpuarray/array.h"
#include "gpuarray/error.h"
#include "gpuarray/types.h"

extern void *ctx;

void setup(void);
void teardown(void);

#define ga_assert_ok(e) ck_assert_int_eq(e, GA_NO_ERROR)

START_TEST(test_take1_ok) {
  GpuArray base;
  GpuArray idx;
  GpuArray res;
  GpuArray v;
  GpuArray vidx;
  GpuArray vres;
  static const uint32_t data[24] = { 0,  1,  2,  3,  4,  5,
                                     6,  7,  8,  9, 10, 11,
                                    12, 13, 14, 15, 16, 17,
                                    18, 19, 20, 21, 22, 23};
  uint32_t buf[12 * 24];
  static const size_t data_dims[1] = {24};
  ssize_t indexes[12];
  size_t dims[3];

  ga_assert_ok(GpuArray_empty(&base, ctx, GA_UINT, 1, data_dims, GA_C_ORDER));
  ga_assert_ok(GpuArray_write(&base, data, sizeof(data)));
  dims[0] = 12;
  ga_assert_ok(GpuArray_empty(&idx, ctx, GA_SSIZE, 1, dims, GA_C_ORDER));
  dims[1] = 6;
  ga_assert_ok(GpuArray_empty(&res, ctx, GA_UINT, 2, dims, GA_C_ORDER));

  /* test v[[1, 0]] on 1d (4) */
  indexes[0] = 1;
  indexes[1] = 0;
  ga_assert_ok(GpuArray_write(&idx, indexes, sizeof(ssize_t) * 2));

  ga_assert_ok(GpuArray_view(&v, &base));
  ga_assert_ok(GpuArray_view(&vidx, &idx));
  ga_assert_ok(GpuArray_view(&vres, &res));

  v.dimensions[0] = 4;

  vidx.dimensions[0] = 2;

  vres.nd = 1;
  vres.dimensions[0] = vidx.dimensions[0];
  vres.strides[0] = v.strides[0];

  ga_assert_ok(GpuArray_take1(&vres, &v, &vidx, 0));
  ga_assert_ok(GpuArray_read(buf, sizeof(uint32_t) * 2, &vres));
  ck_assert(buf[0] == 1);
  ck_assert(buf[1] == 0);

  /* test v[[2, 3, -1]] on 2d (4, 5) */

  GpuArray_clear(&v);
  GpuArray_clear(&vidx);
  GpuArray_clear(&vres);

  indexes[0] = 2;
  indexes[1] = 3;
  indexes[2] = -1;
  ga_assert_ok(GpuArray_write(&idx, indexes, sizeof(ssize_t) * 3));

  ga_assert_ok(GpuArray_view(&v, &base));
  ga_assert_ok(GpuArray_view(&vidx, &idx));
  ga_assert_ok(GpuArray_view(&vres, &res));

  vidx.dimensions[0] = 3;

  dims[0] = 4;
  dims[1] = 6;
  ga_assert_ok(GpuArray_reshape_inplace(&v, 2, dims, GA_ANY_ORDER));
  v.dimensions[1] = 5;
  v.strides[0] = v.dimensions[1] * v.strides[1];

  dims[0] = 3;
  dims[1] = 24;
  ga_assert_ok(GpuArray_reshape_inplace(&vres, 2, dims, GA_C_ORDER));
  vres.dimensions[1] = v.dimensions[1];
  vres.strides[0] = v.strides[0];

  ga_assert_ok(GpuArray_take1(&vres, &v, &vidx, 0));
  ga_assert_ok(GpuArray_read(buf, sizeof(uint32_t) * 15, &vres));

  ck_assert(buf[0] == 10);
  ck_assert(buf[1] == 11);
  ck_assert(buf[2] == 12);
  ck_assert(buf[3] == 13);
  ck_assert(buf[4] == 14);

  ck_assert(buf[5] == 15);
  ck_assert(buf[6] == 16);
  ck_assert(buf[7] == 17);
  ck_assert(buf[8] == 18);
  ck_assert(buf[9] == 19);

  ck_assert(buf[10] == 15);
  ck_assert(buf[11] == 16);
  ck_assert(buf[12] == 17);
  ck_assert(buf[13] == 18);
  ck_assert(buf[14] == 19);

  /* test v[[3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]] on 3d */
  GpuArray_clear(&v);
  GpuArray_clear(&vidx);
  GpuArray_clear(&vres);

  indexes[0] = 3;
  indexes[1] = 3;
  indexes[2] = 1;
  indexes[3] = 1;
  indexes[4] = 2;
  indexes[5] = 2;
  indexes[6] = 0;
  indexes[7] = 0;
  indexes[8] = -1;
  indexes[9] = -2;
  indexes[10] = -3;
  indexes[11] = -4;
  ga_assert_ok(GpuArray_write(&idx, indexes, sizeof(indexes)));

  ga_assert_ok(GpuArray_view(&v, &base));
  ga_assert_ok(GpuArray_view(&vidx, &idx));
  ga_assert_ok(GpuArray_view(&vres, &res));

  dims[0] = 4;
  dims[1] = 2;
  dims[2] = 3;
  ga_assert_ok(GpuArray_reshape_inplace(&v, 3, dims, GA_ANY_ORDER));

  dims[0] = 12;
  dims[1] = 2;
  dims[2] = 3;
  ga_assert_ok(GpuArray_reshape_inplace(&vres, 3, dims, GA_C_ORDER));

  ga_assert_ok(GpuArray_take1(&vres, &v, &vidx, 0));
  ga_assert_ok(GpuArray_read(buf, sizeof(uint32_t) * 72, &vres));

  /* 0 */
  ck_assert(buf[0] == 18);
  ck_assert(buf[1] == 19);
  ck_assert(buf[2] == 20);
  ck_assert(buf[3] == 21);
  ck_assert(buf[4] == 22);
  ck_assert(buf[5] == 23);

  /* 1 */
  ck_assert(buf[6] == 18);
  ck_assert(buf[7] == 19);
  ck_assert(buf[8] == 20);
  ck_assert(buf[9] == 21);
  ck_assert(buf[10] == 22);
  ck_assert(buf[11] == 23);

  /* 2 */
  ck_assert(buf[12] == 6);
  ck_assert(buf[13] == 7);
  ck_assert(buf[14] == 8);
  ck_assert(buf[15] == 9);
  ck_assert(buf[16] == 10);
  ck_assert(buf[17] == 11);

  /* 3 */
  ck_assert(buf[18] == 6);
  ck_assert(buf[19] == 7);
  ck_assert(buf[20] == 8);
  ck_assert(buf[21] == 9);
  ck_assert(buf[22] == 10);
  ck_assert(buf[23] == 11);

  /* 4 */
  ck_assert(buf[24] == 12);
  ck_assert(buf[25] == 13);
  ck_assert(buf[26] == 14);
  ck_assert(buf[27] == 15);
  ck_assert(buf[28] == 16);
  ck_assert(buf[29] == 17);

  /* 5 */
  ck_assert(buf[30] == 12);
  ck_assert(buf[31] == 13);
  ck_assert(buf[32] == 14);
  ck_assert(buf[33] == 15);
  ck_assert(buf[34] == 16);
  ck_assert(buf[35] == 17);

  /* 6 */
  ck_assert(buf[36] == 0);
  ck_assert(buf[37] == 1);
  ck_assert(buf[38] == 2);
  ck_assert(buf[39] == 3);
  ck_assert(buf[40] == 4);
  ck_assert(buf[41] == 5);

  /* 7 */
  ck_assert(buf[42] == 0);
  ck_assert(buf[43] == 1);
  ck_assert(buf[44] == 2);
  ck_assert(buf[45] == 3);
  ck_assert(buf[46] == 4);
  ck_assert(buf[47] == 5);

  /* 8 */
  ck_assert(buf[48] == 18);
  ck_assert(buf[49] == 19);
  ck_assert(buf[50] == 20);
  ck_assert(buf[51] == 21);
  ck_assert(buf[52] == 22);
  ck_assert(buf[53] == 23);

  /* 9 */
  ck_assert(buf[54] == 12);
  ck_assert(buf[55] == 13);
  ck_assert(buf[56] == 14);
  ck_assert(buf[57] == 15);
  ck_assert(buf[58] == 16);
  ck_assert(buf[59] == 17);

  /* 10 */
  ck_assert(buf[60] == 6);
  ck_assert(buf[61] == 7);
  ck_assert(buf[62] == 8);
  ck_assert(buf[63] == 9);
  ck_assert(buf[64] == 10);
  ck_assert(buf[65] == 11);

  /* 11 */
  ck_assert(buf[66] == 0);
  ck_assert(buf[67] == 1);
  ck_assert(buf[68] == 2);
  ck_assert(buf[69] == 3);
  ck_assert(buf[70] == 4);
  ck_assert(buf[71] == 5);
}
END_TEST

Suite *get_suite(void) {
  Suite *s = suite_create("array");
  TCase *tc = tcase_create("take1");
  tcase_add_checked_fixture(tc, setup, teardown);
  tcase_set_timeout(tc, 8.0);
  tcase_add_test(tc, test_take1_ok);
  suite_add_tcase(s, tc);
  return s;
}
