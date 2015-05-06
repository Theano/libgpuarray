/*
 * But it allow faster conversion to this new library of existing code
 */
#ifndef GPUARRAY_NUMPY_COMPAT
#define GPUARRAY_NUMPY_COMPAT

int PyGpuArray_NDIM(const PyGpuArrayObject *arr) {
  return arr->ga.nd;
}
const size_t *PyGpuArray_DIMS(const PyGpuArrayObject *arr) {
  return arr->ga.dimensions;
}

const ssize_t *PyGpuArray_STRIDES(const PyGpuArrayObject* arr) {
  return arr->ga.strides;
}
size_t PyGpuArray_DIM(const PyGpuArrayObject* arr, int n) {
  return arr->ga.dimensions[n];
}
ssize_t PyGpuArray_STRIDE(const PyGpuArrayObject* arr, int n) {
  return arr->ga.strides[n];
}
size_t PyGpuArray_SIZE(const PyGpuArrayObject* arr) {
  size_t size = 1;
  for(int i=0; i< arr->ga.nd; i++) {
    size *= arr->ga.dimensions[i];
  }
  return size;
}

#endif
