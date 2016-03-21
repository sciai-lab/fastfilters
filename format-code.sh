#!/bin/sh
for f in `find src tests include -type f -iname \*.\?xx`; do
	clang-format -i $f
done