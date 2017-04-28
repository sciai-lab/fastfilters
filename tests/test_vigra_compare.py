from __future__ import print_function

import sys
print("\nexecuting test file", __file__, file=sys.stderr)
exec(compile(open('set_paths.py', "rb").read(), 'set_paths.py', 'exec'))
import fastfilters as ff
import numpy as np
import vigra

def test_vigra_compare():
    a = np.random.randn(1000000).reshape(1000,1000).astype(np.float32)[:,:900]
    a = np.ascontiguousarray(a)

    sigmas = [1.0, 5.0, 10.0]

    for order in [0,1,2]:
        for sigma in sigmas:
            res_ff = ff.gaussianDerivative(a, sigma, order, window_size=3.5)
            res_vigra = vigra.filters.gaussianDerivative(a, sigma, [order,order], window_size=3.5)

            print("gaussian ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

            if not np.allclose(res_ff, res_vigra, atol=1e-6):
                raise Exception("FAIL: ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.hessianOfGaussianEigenvalues(a, sigma, window_size=3.5)
        res_vigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma, window_size=3.5)
        print("HOG", sigma, np.max(np.abs(res_ff - res_vigra)))

        if not np.allclose(res_ff, res_vigra, atol=1e-6) or np.any(np.isnan(np.abs(res_ff - res_vigra))):
            raise Exception("FAIL: HOG", sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.gaussianGradientMagnitude(a, sigma, window_size=3.5)
        res_vigra = vigra.filters.gaussianGradientMagnitude(a, sigma, window_size=3.5)
        print("gradmag2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            raise Exception("FAIL: gradmag2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.laplacianOfGaussian(a, sigma, window_size=3.5)
        res_vigra = vigra.filters.laplacianOfGaussian(a, sigma, window_size=3.5)
        print("laplacian2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            raise Exception("FAIL: laplacian2d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        for sigma2 in sigmas:
            res_ff = ff.structureTensorEigenvalues(a, sigma2, sigma, window_size=3.5)
            res_vigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2, window_size=3.5)
            print("ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))

            if not np.allclose(res_ff, res_vigra, atol=1e-6) or np.any(np.isnan(np.abs(res_ff - res_vigra))):
                raise Exception("FAIL: ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))
