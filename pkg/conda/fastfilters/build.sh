#!/bin/bash

./waf configure --prefix=${PREFIX} --python=${PYTHON}
./waf
./waf install

mkdir -p ${PREFIX}/lib
mv ${PREFIX}/lib64/libfastfilters.so ${PREFIX}/lib/
