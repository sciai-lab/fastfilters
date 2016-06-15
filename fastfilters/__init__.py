from .core import *
__all__ = ["g"]
__version__ = core.__version__

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

def gaussianSmoothing(array, sigma, window_size=0.0):
	return __get_fn(array, gaussian2d, gaussian3d)(array, 0, sigma, window_size=window_size)