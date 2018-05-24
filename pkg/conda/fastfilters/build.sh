#!/bin/bash

if [ $(uname) == Darwin ]; then
    CC=clang
    CXX=clang++
    CXXFLAGS="-stdlib=libc++"
else
    CC=gcc
    CXX=g++
    # enable compilation without CXX abi to stay compatible with gcc < 5 built packages
    if [[ ${DO_NOT_BUILD_WITH_CXX11_ABI} == '1' ]]; then
        CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 ${CXXFLAGS}"
    fi
fi

mkdir build_conda
cd build_conda
cmake \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
    ..
make -j${CPU_COUNT}
#make fastfilters_py_test
make install
