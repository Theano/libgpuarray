#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <check.h>

#include "gpuarray/buffer.h"
#include "gpuarray/error.h"

char* dev_name = NULL;

int get_env_dev(const char **name, gpucontext_props *p) {
  char *dev = NULL;
  char *end;
  long no;
  int pl;
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
    gpucontext_props_cuda_dev(p, (int)no);
    return 0;
  }
  if (strncmp(dev, "opencl", 6) == 0) {
    *name = "opencl";
    no = strtol(dev + 6, &end, 10);
    if (end == dev || *end != ':')
      return -1;
    if (no < 0 || no > 32768)
      return -1;
    pl = (int)no;
    dev = end;
    no = strtol(dev + 1, &end, 10);
    if (end == dev || *end != '\0')
      return -1;
    if (no < 0 || no > 32768)
      return -1;
    gpucontext_props_opencl_dev(p, pl, (int)no);
    return 0;
  }
  return -1;
}

gpucontext *ctx;

void setup(void) {
  const char *name = NULL;
  gpucontext_props *p;
  ck_assert_int_eq(gpucontext_props_new(&p), GA_NO_ERROR);
  ck_assert_int_eq(get_env_dev(&name, p), 0);
  ck_assert_int_eq(gpucontext_init(&ctx, name, p), GA_NO_ERROR);
  ck_assert_ptr_ne(ctx, NULL);
}

void teardown(void) {
  gpucontext_deref(ctx);
  ctx = NULL;
}
