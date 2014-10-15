Goal
====

Make a common GPU ndarray(n dimensions array) that can be reused by
all projects that is as future proof as possible, while keeping it easy
to use for simple need/quick test.


Motivation
----------

* Currently there are at least 6 different GPU arrays in python
    * CudaNdarray(Theano), GPUArray(pycuda), CUDAMatrix(cudamat), GPUArray(pyopencl), Clyther, Copperhead, ...
    * There are even more if we include other languages.
* They are incompatible
    * None have the same properties and interface
*   All of them are a subset of NumPy.ndarray on the GPU!


Design Goals
------------

* Have a n dimensional array.
    * Otherwise, not all project can reuse it. And you never know when you will need more dimensions.
* Support many data types (int, float, double).
    * Otherwise, we are limited in what we can do with it.
* Support strided view, c and f memory layout
    * This lowers memory usage and memory copies. A scarce resource on GPU.
    * You never know which memory layout is the best for your future need.
* Be compatible with CUDA and OpenCL
    * You never know the future. Also, this make it possible to support other future language.
* Make it easy to support just a subset of the feature.
    * If you just want to test something that support only CUDA and c contiguous matrices, it will stay easy as without libgpuarray.
    * There is functionality to make the same code work and compile with both CUDA and OpenCL. You don't need to use them.
* Have the base object in C to allow collaboration with more projects.
    * We want people from C, C++, ruby, R, ... all use the same base GPU ndarray.
* Have a python binding separate from the c code.
* Support mixed back-end OpenCL/CUDA in the same binary.
    * But still keep it easy to use only one.
    * This would allow an easier transition to a new platform if the need come.
* Support dynamic compilation
    * This allow optimization at run time based on the shapes for example.
    * You don't need to use this.

In the end, we need a NumPy ndarray on the GPU! There is a restriction
that does not allow us to reuse that object directly, but you will find
it very similar.
