#!/bin/bash

if [ $(uname) == Darwin ]; then
    CC=clang
    CXX=clang++
    CXXFLAGS="-stdlib=libc++"
else
    CC=gcc
    CXX=g++
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 ${CXXFLAGS}"
fi

mkdir build_conda
cd build_conda
cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} ..
make -j${CPU_COUNT}
#make fastfilters_py_test
make install
