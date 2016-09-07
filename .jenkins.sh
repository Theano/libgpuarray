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

# Set some default values
: ${BUILDBOT_DIR:="$WORKSPACE/nightly_build"} # Jenkins workspace path
# Can also set to "Debug", "Release" to go faster
: ${GPUARRAY_CONFIG:="Release"}
# Set these to " " to disable (empty doesn't work)
: ${DEVICES_CUDA:="cuda"} # for multiple devices use "cuda0 cuda1"
: ${DEVICES_OPENCL:=" "}
# Parameters for nosetests
: ${NOSE_PARAM="-v --with-xunit --xunit-file="}

mkdir -p ${BUILDBOT_DIR}
cd ${BUILDBOT_DIR}

# Make fresh clone (with no history since we don't need it)
rm -rf libgpuarray
git clone --depth 1 "https://github.com/Theano/libgpuarray.git"

(cd libgpuarray && echo "libgpuarray commit" && git rev-parse HEAD)

# Clean up previous installs (to make sure no old files are left)
rm -rf local
mkdir local

# Build libgpuarray and run C tests
mkdir libgpuarray/build
(cd libgpuarray/build && cmake .. -DCMAKE_BUILD_TYPE=${GPUARRAY_CONFIG} -DCMAKE_INSTALL_PREFIX=${BUILDBOT_DIR}/local && make)

# Test on different devices
for dev in ${DEVICES_CUDA}; do
    echo "Testing libgpuarray for DEVICE=${dev}"
    (cd libgpuarray/build && CK_DEFAULT_TIMEOUT=16 DEVICE=${dev} make test)
done
for dev in ${DEVICES_OPENCL}; do
    echo "Testing libgpuarray for DEVICE=${dev}"
    (cd libgpuarray/build && DEVICE=${dev} make test)
done

# Finally install
(cd libgpuarray/build && make install)
export LD_LIBRARY_PATH=${BUILDBOT_DIR}/local/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${BUILDBOT_DIR}/local/lib:${LIBRARY_PATH}
export CPATH=${BUILDBOT_DIR}/local/include:${CPATH}

# Build the pygpu modules
(cd libgpuarray && python setup.py build_ext --inplace -I${BUILDBOT_DIR}/local/include -L${BUILDBOT_DIR}/local/lib)

# Test it
for dev in ${DEVICES_CUDA}; do
    echo "Testing pygpu for DEVICE=${dev}"
    test=${BUILDBOT_DIR}/pygpu
    DEVICE=${dev} time nosetests --with-xunit --xunit-file=${test}${dev}tests.xml libgpuarray/pygpu/tests
done
for dev in ${DEVICES_OPENCL}; do
    echo "Testing pygpu for DEVICE=${dev}"
    DEVICE=${dev} time nosetests --with-xunit --xunit-file=${test}${dev}tests.xml libgpuarray/pygpu/tests -e test_blas.py
done
