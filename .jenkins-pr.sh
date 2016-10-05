#!/bin/bash

# Script for Jenkins continuous integration testing of libgpuarray

# Print commands as they are executed
set -x

# Anaconda python
export PATH=/usr/local/miniconda2/bin:$PATH

# CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

# Can also set to "Debug", "Release" to go faster
: ${GPUARRAY_CONFIG:="Release"}
# Set these to " " to disable (empty doesn't work)
: ${DEVICES_CUDA:="cuda"} # for multiple devices use "cuda0 cuda1"
: ${DEVICES_OPENCL:=" "}

git rev-parse HEAD

# Build libgpuarray and run C tests
mkdir build
(cd build && cmake .. -DCMAKE_BUILD_TYPE=${GPUARRAY_CONFIG} && make)

# Test on different devices
for dev in ${DEVICES_CUDA}; do
    echo "Testing libgpuarray for DEVICE=${dev}"
    (cd build && DEVICE=${dev} make test)
done
for dev in ${DEVICES_OPENCL}; do
    echo "Testing libgpuarray for DEVICE=${dev}"
    (cd build && DEVICE=${dev} make test)
done

export LD_LIBRARY_PATH=`pwd`/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=`pwd`/lib:${LIBRARY_PATH}
export CPATH=`pwd`/src:${CPATH}

# Build the pygpu modules
python setup.py build_ext --inplace

# Test it
test=pygpu
for dev in ${DEVICES_CUDA}; do
    echo "Testing pygpu for DEVICE=${dev}"
    DEVICE=${dev} time nosetests --with-xunit --xunit-file=${test}${dev}tests.xml pygpu/tests
done
for dev in ${DEVICES_OPENCL}; do
    echo "Testing pygpu for DEVICE=${dev}"
    DEVICE=${dev} time nosetests --with-xunit --xunit-file=${test}${dev}tests.xml pygpu/tests -e test_blas.py
done
