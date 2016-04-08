#!/bin/sh
for f in `find src include -type f -iname \*.\?xx`; do
	clang-format -i $f
done
for f in `find src include -type f -iname \*.c`; do
	clang-format -i $f
done
for f in `find src include -type f -iname \*.h`; do
	clang-format -i $f
done