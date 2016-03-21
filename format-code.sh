#!/bin/sh
for f in `find src tests include -type f`; do
	clang-format -i $f
done