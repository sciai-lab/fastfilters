from .core import *
import numpy as np

__version__ = core.__version__

def __reorder_array(array):
	if array.flags['C_CONTIGUOUS']: return array

	array = array.transpose(np.argsort(array.strides)[::-1])
	return np.ascontiguousarray(array)

def __p_fix_array(func):
	def func_wrapper(array, *args, **kwargs):
		return func(__reorder_array(array), *args, **kwargs)
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