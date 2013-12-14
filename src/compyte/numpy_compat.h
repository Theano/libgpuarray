/*
 * This file isn't recommanded. Using it make your code not able to work on OpenCL.
 *
 * But it allow faster conversion to this new library of existing code
 */
#include "extension.h"
#ifndef COMPYTE_BUFFER_CUDA_H
#define COMPYTE_BUFFER_CUDA_H

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
//int PyGpuArray_ITEMSIZE(PyGpuArrayObject* arr)
size_t PyGpuArray_SIZE(const PyGpuArrayObject* arr) {
  size_t size = 1;
  for(int i=0; i< arr->ga.nd; i++) {
    size *= arr->ga.dimensions[i];
  }
  return size;
}

#endif
