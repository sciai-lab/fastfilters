from __future__ import absolute_import
from . import core
import numpy as np

__all__ = ["gaussianSmoothing", "gaussianGradientMagnitude", "hessianOfGaussianEigenvalues", "laplacianOfGaussian", "structureTensorEigenvalues", "gaussianDerivative"]
__version__ = core.__version__

try:
	import vigra
except ImportError:
	pass

def __p_fix_array(func):
	"""
	Decorator.
	Remove singleton dimensions from the input array before calling the wrapped function,
	then "unsqueeze" the result so it corresponds to the shape of the input data.

	Note: Singleton dimensions are only permitted if the input array has axistags.
		  Otherwise, there is no way to know which singleton dimensions (if any) correspond to channel.
	"""
	def func_wrapper(array, *args, **kwargs):
		if hasattr(array, 'axistags'):
			array = vigra.taggedView( np.ascontiguousarray(array), array.axistags )
			squeezed = array.squeeze()
			res = func(squeezed, *args, **kwargs)

			if res.shape == squeezed.shape:
				res = vigra.taggedView( res, squeezed.axistags )
			else:
				res = vigra.taggedView( res, list(squeezed.axistags) + [vigra.AxisInfo.c] )
			return res.withAxes(array.axistags)
		else:
			assert not any( np.array(array.shape) == 1 ), \
				"Can't handle arrays with singleton dimensions (unless they are tagged VigraArrays)."
			return func(array, *args, **kwargs)

	return func_wrapper

def __get_fn(array, fn_2d, fn_3d):
	"""
	Decide whether or not the given array is really 2D or 3D, and return the corresponding function.
	"""
	if hasattr(array, 'axistags'):
		assert array.channels == 1, \
			"Can't handle multi-channel data. " \
			"(Your image has {} channels.)".format(array.channels)

		time_index = array.axistags.index('t')
		assert time_index == len(array.axistags) or array.shape[time_index] == 1 , \
			"Can't handle arrays with multiple time steps. " \
			"(Your image has {})".format(array.shape[time_index])

		assert array.squeeze().ndim in (2,3), \
			"Invalid array axes/dimensions: '{}'/{}".format(array.axistags, array.shape)

	squeezed = array.squeeze()
	if squeezed.ndim == 2:
		return fn_2d
	elif squeezed.ndim == 3:
		return fn_3d
	else:
		raise NotImplementedError("Invalid array dimensions: {}".format(  array.shape ))

@__p_fix_array
def gaussianSmoothing(array, sigma, window_size=0.0):
	return __get_fn(array, core.gaussian2d, core.gaussian3d)(array, 0, sigma, window_size)

@__p_fix_array
def gaussianGradientMagnitude(array, sigma, window_size=0.0):
	return __get_fn(array, core.gradmag2d, core.gradmag3d)(array, sigma, window_size)

@__p_fix_array
def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
	res = __get_fn(image, core.hog2d, core.hog3d)(image, scale, window_size)
	return np.rollaxis(res, 0, len(res.shape))

@__p_fix_array
def laplacianOfGaussian(array, scale=1.0, window_size=0.0):
	return __get_fn(array, core.laplacian2d, core.laplacian3d)(array, scale, window_size)

@__p_fix_array
def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
	res = __get_fn(image, core.st2d, core.st3d)(image, innerScale, outerScale, window_size)
	return np.rollaxis(res, 0, len(res.shape))

@__p_fix_array
def gaussianDerivative(array, sigma, order, window_size=0.0):
    if isinstance(order, list):
        assert(len(order) == len(array.shape))
        assert(len(np.unique(order)) == 1)
        order = order[0]
    return __get_fn(array, core.gaussian2d, core.gaussian3d)(array, order, sigma, window_size)
