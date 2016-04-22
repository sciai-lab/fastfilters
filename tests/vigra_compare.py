import fastfilters as ff
import numpy as np
import sys

try:
	import vigra
except ImportError:
	print("WARNING: vigra not available - skipping tests.")
	with open(sys.argv[1], 'w') as f:
		f.write('')
	exit()

a = np.random.randn(1000000).reshape(1000,1000)


for order in [0,1,2]:
	for sigma in [1.0, 5.0, 10.0]:
		res_ff = ff.gaussian2d(a, order, sigma)
		res_vigra = vigra.filters.gaussianDerivative(a, sigma, [order,order])

		if not np.allclose(res_ff, res_vigra, atol=1e-6):
			print(order, sigma, np.max(np.abs(res_ff - res_vigra)))
			raise Exception()


np.unique(ff.hog2d(a, 1.0))