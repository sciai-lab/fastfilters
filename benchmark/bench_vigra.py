import numpy as np
import fastfilters as ff
import vigra
import time


class Timer(object):
	def __enter__(self):
		self.a = time.clock()
		return self

	def __exit__(self, *args):
		self.b = time.clock()
		self.delta = self.b - self.a

a = np.zeros((5000,5000)).astype(np.float32)

for order in [0,1,2]:
	for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianDerivative(a, sigma, [order, order])

		with Timer() as tff:
			resff = ff.gaussianDerivative(a, sigma, [order, order])

		fact = tvigra.delta / tff.delta

		print("Timing gaussian 2D with order = %d and sigma = %d:  vigra = %f, ff = %f --> speedup: %f" % (order, sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianGradientMagnitude(a, sigma)

		with Timer() as tff:
			resff = ff.gaussianGradientMagnitude(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing gradient magnitude 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))

for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.laplacianOfGaussian(a, sigma)

		with Timer() as tff:
			resff = ff.laplacianOfGaussian(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing laplacian 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))



for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma)

		with Timer() as tff:
			resff = ff.hessianOfGaussianEigenvalues(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing HOG 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5,6,7,8,9,10]:
	sigma2 = 2*sigma
	with Timer() as tvigra:
		resvigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2)

	with Timer() as tff:
		resff = ff.structureTensorEigenvalues(a, sigma, sigma2)

	fact = tvigra.delta / tff.delta

	print("Timing ST 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))


a = np.zeros((100,100,100)).astype(np.float32)

for order in [0,1,2]:
	for sigma in [1,2,3,4,5]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianDerivative(a, sigma, [order, order, order])

		with Timer() as tff:
			resff = ff.gaussianDerivative(a, sigma, [order, order, order])

		fact = tvigra.delta / tff.delta

		print("Timing gaussian 3D with order = %d and sigma = %d:  vigra = %f, ff = %f --> speedup: %f" % (order, sigma, tvigra.delta, tff.delta, fact))

for sigma in [1,2,3,4,5]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianGradientMagnitude(a, sigma)

		with Timer() as tff:
			resff = ff.gaussianGradientMagnitude(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing gradient magnitude 3D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5]:
		with Timer() as tvigra:
			resvigra = vigra.filters.laplacianOfGaussian(a, sigma)

		with Timer() as tff:
			resff = ff.laplacianOfGaussian(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing laplacian 3D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))

for sigma in [1,2,3,4,5]:
		with Timer() as tvigra:
			resvigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma)

		with Timer() as tff:
			resff = ff.hessianOfGaussianEigenvalues(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing HOG 3D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5]:
	sigma2 = 2*sigma
	with Timer() as tvigra:
		resvigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2)

	with Timer() as tff:
		resff = ff.structureTensorEigenvalues(a, sigma, sigma2)

	fact = tvigra.delta / tff.delta

	print("Timing ST 3D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))