from __future__ import print_function

import sys
print("\nexecuting test file testtest", __file__, file=sys.stderr)
exec(compile(open('set_paths.py', "rb").read(), 'set_paths.py', 'exec'))
import fastfilters as ff
import numpy as np
from nose.tools import ok_

def test_border_bug():
    a = np.ones((150,150), dtype=np.float32)
    a[:] = np.nan
    a[25:125, 25:125] = 0

    for _ in range(10):
        res = ff.gaussianSmoothing(a[25:125, 25:125], np.sqrt(99), window_size=10)
        ok_(np.all(res == 0))
