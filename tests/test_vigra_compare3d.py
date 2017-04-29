from __future__ import print_function

import sys
print("\nexecuting test file", __file__, file=sys.stderr)
exec(compile(open('set_paths.py', "rb").read(), 'set_paths.py', 'exec'))
import fastfilters as ff
import numpy as np
import vigra

def test_vigra_compare3d():
    a = np.random.randn(1000000).reshape(100,100,100).astype(np.float32)[:,:90,:80]
    a = np.ascontiguousarray(a)

    sigmas = [1.0, 5.0, 10.0]

    for order in [0,1,2]:
        for sigma in sigmas:
            res_ff = ff.gaussianDerivative(a, sigma, order)
            res_vigra = vigra.filters.gaussianDerivative(a, sigma, [order,order,order])

            print("gaussian ", order, sigma, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

            if not np.allclose(res_ff, res_vigra, atol=1e-6):
                raise Exception("FAIL: ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.gaussianGradientMagnitude(a, sigma)
        res_vigra = vigra.filters.gaussianGradientMagnitude(a, sigma)
        print("gradmag3d ", order, sigma, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            raise Exception("FAIL: gradmag3d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

    for sigma in sigmas:
        res_ff = ff.laplacianOfGaussian(a, sigma)
        res_vigra = vigra.filters.laplacianOfGaussian(a, sigma)
        print("laplacian3d ", order, sigma, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            raise Exception("FAIL: laplacian3d ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

    for sigma in sigmas:
        res_ff = ff.hessianOfGaussianEigenvalues(a, sigma)
        res_vigra = vigra.filters.hessianOfGaussianEigenvalues(a, sigma)
        print("HOG", sigma, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

        if np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra) > 1e-6 or np.any(np.isnan(np.abs(res_ff - res_vigra))):
            raise Exception("FAIL: HOG", sigma, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

    for sigma in sigmas:
        for sigma2 in sigmas:
            res_ff = ff.structureTensorEigenvalues(a, sigma2, sigma)
            res_vigra = vigra.filters.structureTensorEigenvalues(a, sigma, sigma2)
            print("ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))

            if np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra) > 1e-5 or np.any(np.isnan(np.abs(res_ff - res_vigra))):
                raise Exception("FAIL: ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)), np.sum(np.abs(res_ff-res_vigra))/np.size(res_vigra))
