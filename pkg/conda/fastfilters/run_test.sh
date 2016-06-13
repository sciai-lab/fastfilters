python ${SRC_DIR}/tests/vigra_compare.py compare-success.txt
test -f compare-success.txt

python ${SRC_DIR}/tests/vigra_compare3d.py compare-3d-success.txt
test -f compare-3d-success.txt

python ${SRC_DIR}/tests/vigra_compare_rgb.py compare-rgb-success.txt
test -f compare-rgb-success.txt

