import fastfilters as ff
import numpy as np

a = np.ones((150,150), dtype=np.float32)
a[:] = np.nan
a[25:125, 25:125] = 1

for _ in range(10):
	res = ff.gaussianSmoothing(a[25:125, 25:125], np.sqrt(99), window_size=5)
	assert(np.any(np.isnan(res)) == False)
	assert(np.any(np.where(res < -1000)) == False)