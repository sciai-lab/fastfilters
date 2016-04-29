#!/bin/bash

./waf configure --prefix=${PREFIX} --python=${PYTHON}
./waf
./waf install