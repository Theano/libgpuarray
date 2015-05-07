/*
 * But it allow faster conversion to this new library of existing code
 */
#ifndef GPUARRAY_NUMPY_COMPAT
#define GPUARRAY_NUMPY_COMPAT

static int PyGpuArray_NDIM(const PyGpuArrayObject *arr) {
  return arr->ga.nd;
}

static const size_t *PyGpuArray_DIMS(const PyGpuArrayObject *arr) {
  return arr->ga.dimensions;
}

static const ssize_t *PyGpuArray_STRIDES(const PyGpuArrayObject* arr) {
  return arr->ga.strides;
}

static size_t PyGpuArray_DIM(const PyGpuArrayObject* arr, int n) {
  return arr->ga.dimensions[n];
}

static ssize_t PyGpuArray_STRIDE(const PyGpuArrayObject* arr, int n) {
  return arr->ga.strides[n];
}

static size_t PyGpuArray_SIZE(const PyGpuArrayObject* arr) {
  size_t size = 1;
  for(int i=0; i< arr->ga.nd; i++) {
    size *= arr->ga.dimensions[i];
  }
  return size;
}

#endif
