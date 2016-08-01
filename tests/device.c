#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <check.h>

#include "gpuarray/buffer.h"

char* dev_name = NULL;

int get_env_dev(const char **name) {
  char *dev = NULL;
  char *end;
  long no;
  int d;
  dev = dev_name;
  if (dev == NULL) {
    if ((dev = getenv("GPUARRAY_TEST_DEVICE")) == NULL) {
      if ((dev = getenv("DEVICE")) == NULL) {
        fprintf(stderr, "No device specified for testing, specify a device with DEVICE or GPUARRAY_TEST_DEVICE");
        return -1;
      }
    }
  }
  if (strncmp(dev, "cuda", 4) == 0) {
    *name = "cuda";
    no = strtol(dev + 4, &end, 10);
    if (end == dev || *end != '\0')
      return -1;
    if (no < 0 || no > INT_MAX)
      return -1;
    return (int)no;
  }
  if (strncmp(dev, "opencl", 6) == 0) {
    *name = "opencl";
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

gpucontext *ctx;

void setup(void) {
  const char *name = NULL;
  int dev = get_env_dev(&name);
  if (dev == -1)
    ck_abort_msg("Bad test device");
  ctx = gpucontext_init(name, dev, 0, NULL);
  ck_assert_ptr_ne(ctx, NULL);
}

void teardown(void) {
  gpucontext_deref(ctx);
  ctx = NULL;
}
