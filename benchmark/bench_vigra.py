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

a = np.zeros((1000,1000))

for order in [0,1,2]:
	for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianDerivative(a, sigma, [order, order])

		with Timer() as tff:
			resff = ff.gaussian2d(a, int(order), float(sigma))

		fact = tvigra.delta / tff.delta

		print("Timing gaussian 2D with order = %d and sigma = %d:  vigra = %f, ff = %f --> speedup: %f" % (order, sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.gaussianGradientMagnitude(a, sigma)

		with Timer() as tff:
			resff = ff.gradmag2d(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing gradient magnitude 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))

for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.laplacianOfGaussian(a, sigma)

		with Timer() as tff:
			resff = ff.laplacian2d(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing laplacian 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))



for sigma in [1,2,3,4,5,6,7,8,9,10]:
		with Timer() as tvigra:
			resvigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma)

		with Timer() as tff:
			resff = ff.hog2d(a, sigma)

		fact = tvigra.delta / tff.delta

		print("Timing HOG 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))


for sigma in [1,2,3,4,5,6,7,8,9,10]:
	sigma2 = 2*sigma
	with Timer() as tvigra:
		resvigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2)

	with Timer() as tff:
		resff = ff.st2d(a, sigma2, sigma)

	fact = tvigra.delta / tff.delta

	print("Timing ST 2D with sigma = %f:  vigra = %f, ff = %f --> speedup: %f" % (sigma, tvigra.delta, tff.delta, fact))