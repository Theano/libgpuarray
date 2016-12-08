#!/bin/bash

CMAKE=$CONDA_PREFIX/bin/cmake

mkdir -p $SRC_DIR/build && cd $SRC_DIR/build
$CMAKE $SRC_DIR -DCMAKE_BUILD_TYPE="Release" -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DMPI_LIBRARY=$MPI_LIBRARY
make && make install
