from __future__ import absolute_import
from .core import __version__, gaussian2d, gaussian3d, gradmag2d, gradmag3d, hog2d, hog3d, laplacian2d, laplacian3d, st2d, st3d
import numpy as np

__all__ = ["gaussianSmoothing", "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues", "laplacianOfGaussian", "structureTensorEigenvalues"]

def __p_fix_array(func):
	def func_wrapper(array, *args, **kwargs):
		permutation = []
		if array.dtype == np.float32 and not array.flags['C_CONTIGUOUS']:
			permutation = np.argsort(array.strides)[::-1]
			array = array.transpose(permutation)

			if array.strides[-1] != array.dtype.itemsize:
				array = np.ascontiguousarray(array)

		res = func(array, *args, **kwargs)

		if len(permutation) > 0:
			res = res.transpose(permutation[::-1])

		return res

	return func_wrapper

def __get_fn(array, fn_2d, fn_3d):
	if hasattr(array, 'axistags'):
		if len(array.shape) == 2: return fn_2d
		elif len(array.shape) == 3:
			if array.axistags[2].isSpatial(): return fn_3d
			elif array.axistags[2].isChannel(): return fn_2d
			else: raise NotImplementedError("Invalid array shape.")
		elif len(array.shape) == 4 and array.axistags[3].isChannel(): return fn_3d
		else: raise NotImplementedError("Invalid array shape.")
	else:
		if len(array.shape) == 2: return fn_2d
		elif len(array.shape) == 3:
			if array.shape[2] > 3: return fn_3d
			else: return fn_2d
		elif len(array.shape) == 4: return fn_3d
		else: return NotImplementedError("Invalid array shape.")
	raise NotImplementedError("Invalid array shape.")

@__p_fix_array
def gaussianSmoothing(array, sigma, window_size=0.0):
	return __get_fn(array, gaussian2d, gaussian3d)(array, 0, sigma, window_size)

@__p_fix_array
def gaussianGradientMagnitude(array, sigma, window_size=0.0):
	return __get_fn(array, gradmag2d, gradmag3d)(array, sigma, window_size)

@__p_fix_array
def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
	res = __get_fn(image, hog2d, hog3d)(image, scale, window_size)
	return np.rollaxis(res, 0, len(res.shape))

@__p_fix_array
def laplacianOfGaussian(array, scale=1.0, window_size=0.0):
	return __get_fn(array, laplacian2d, laplacian3d)(array, scale, window_size)

@__p_fix_array
def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
	res = __get_fn(image, st2d, st3d)(image, innerScale, outerScale, window_size)
	return np.rollaxis(res, 0, len(res.shape))