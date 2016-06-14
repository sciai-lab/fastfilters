#!/bin/bash

mkdir build_conda
cd build_conda
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} ..
make
make install
