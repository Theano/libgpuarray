#!/bin/bash

# Script for Jenkins continuous integration testing of libgpuarray on mac

# Print commands as they are executed
set -x

# Set path for conda and cmake
export PATH="/Users/jenkins/miniconda2/bin:/usr/local/bin:$PATH"

# CUDA
export PATH=/usr/local/cuda/bin:${PATH}
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:${DYLD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}

# Can also set to "Debug", "Release" to go faster
: ${GPUARRAY_CONFIG:="Release"}
# Set these to " " to disable (empty doesn't work)
: ${DEVICES_CUDA:="cuda"} # for multiple devices use "cuda0 cuda1"
: ${DEVICES_OPENCL:=" "}

git rev-parse HEAD

# Build libgpuarray and run C tests
rm -rf build lib
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

export PYTHONPATH=`pwd`/lib/python:$PYTHONPATH
export DYLD_LIBRARY_PATH=`pwd`/lib:${DYLD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=`pwd`/src:${CPLUS_INCLUDE_PATH}

# Build the pygpu modules
python setup.py build_ext --inplace -I`pwd`/src -L`pwd`/lib

# Test it
test=pygpu_pr_mac
for dev in ${DEVICES_CUDA}; do
    echo "Testing pygpu for DEVICE=${dev}"
    DEVICE=${dev} nosetests --with-xunit --xunit-file=${test}_${dev}tests.xml pygpu/tests
done
for dev in ${DEVICES_OPENCL}; do
    echo "Testing pygpu for DEVICE=${dev}"
    DEVICE=${dev} nosetests --with-xunit --xunit-file=${test}_${dev}tests.xml pygpu/tests -e test_blas.py
done
