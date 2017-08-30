#!/bin/bash

if [[ $(uname) == Darwin ]]; then
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_OSX_DEPLOYMENT_TARGET=
else
  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX
fi
cmake --build . --config Release --target all
cmake --build . --config Release --target install
