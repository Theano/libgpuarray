#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <check.h>

#include <gpuarray/buffer.h>

int get_env_dev(const gpuarray_buffer_ops **o) {
  char *dev;
  char *end;
  long no;
  int d;
  if ((dev = getenv("GPUARRAY_TEST_DEVICE")) == NULL) {
    if ((dev = getenv("DEVICE")) == NULL) {
      *o = gpuarray_get_ops("opencl");
      return 0; /* opencl0:0 */
    }
  }
  if (strncmp(dev, "cuda", 4) == 0) {
    *o = gpuarray_get_ops("cuda");
    if (*o == NULL) return -1;
    no = strtol(dev + 4, &end, 10);
    if (end == dev || *end != '\0')
      return -1;
    if (no < 0 || no > INT_MAX)
      return -1;
    return (int)no;
  }
  if (strncmp(dev, "opencl", 6) == 0) {
    *o = gpuarray_get_ops("opencl");
    if (*o == NULL) return -1;
    no = strtol(dev + 6, &end, 10);
    if (end == dev || *end != ':')
      return -1;
    if (no < 0 || no > 32768)
      return -1;
    d = (int)no;
    dev = end;
    no = strtol(dev + 1, &end, 10);
    if (end == dev || *end != '\0')
      return -1;
    if (no < 0 || no > 32768)
      return -1;
    d <<= 16;
    d |= (int)no;
    return d;
  }
  return -1;
}

void *ctx;
const gpuarray_buffer_ops *ops;

void setup(void) {
  int dev = get_env_dev(&ops);
  if (dev == -1)
    ck_abort_msg("Bad test device");
  ctx = ops->buffer_init(dev, 0, NULL);
  ck_assert_ptr_ne(ctx, NULL);
}

void teardown(void) {
  ops->buffer_deinit(ctx);
  ctx = NULL;
  ops = NULL;
}
