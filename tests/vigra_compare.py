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

a = np.random.randn(1000000).reshape(1000,1000).astype(np.float32)[:,:900]
a = np.ascontiguousarray(a)

sigmas = [1.0, 5.0, 10.0]

for order in [0,1,2]:
	for sigma in sigmas:
		res_ff = ff.gaussian2d(a, order, sigma)
		res_vigra = vigra.filters.gaussianDerivative(a, sigma, [order,order])

		print("gaussian ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

		if not np.allclose(res_ff, res_vigra, atol=1e-6):
			raise Exception("FAIL: ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


for sigma in sigmas:
	res_ff = ff.hog2d(a, sigma)
	res_vigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma).swapaxes(0,2).swapaxes(1,2)
	print("HOG", sigma, np.max(np.abs(res_ff - res_vigra)))

	if not np.allclose(res_ff, res_vigra, atol=1e-6) or np.any(np.isnan(np.abs(res_ff - res_vigra))):
		raise Exception("FAIL: HOG", sigma, np.max(np.abs(res_ff - res_vigra)))


for sigma in sigmas:
	res_ff = ff.gradmag2d(a, sigma)
	res_vigra = vigra.filters.gaussianGradientMagnitude(a, sigma)
	print("gradmag2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

	if not np.allclose(res_ff, res_vigra, atol=1e-6):
		raise Exception("FAIL: gradmag2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


for sigma in sigmas:
	res_ff = ff.laplacian2d(a, sigma)
	res_vigra = vigra.filters.laplacianOfGaussian(a, sigma)
	print("laplacian2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

	if not np.allclose(res_ff, res_vigra, atol=1e-6):
		raise Exception("FAIL: laplacian2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


for sigma in sigmas:
	for sigma2 in sigmas:
		res_ff = ff.st2d(a, sigma2, sigma)
		res_vigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2).swapaxes(0,2).swapaxes(1,2)
		print("ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))

		if not np.allclose(res_ff, res_vigra, atol=1e-6) or np.any(np.isnan(np.abs(res_ff - res_vigra))):
			raise Exception("FAIL: ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))

with open(sys.argv[1], 'w') as f:
	f.write('')
