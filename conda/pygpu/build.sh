#!/bin/bash

$PYTHON $SRC_DIR/setup.py build_ext -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib
$CONDA_PREFIX/bin/pip install $SRC_DIR
