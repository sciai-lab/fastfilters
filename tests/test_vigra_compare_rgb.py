import sys
print("\nexecuting test file", __file__, file=sys.stderr)
exec(compile(open('set_paths.py', "rb").read(), 'set_paths.py', 'exec'))
import fastfilters.core as ff
import numpy as np

def test_vigra_compare_rgb():
    try:
        import vigra
    except ImportError:
        print("WARNING: vigra not available - skipping tests.")
        with open(sys.argv[1], 'w') as f:
            f.write('')
        exit()

    a = np.random.randn(3000000).reshape(1000,1000,3).astype(np.float32)
    avigra = vigra.RGBImage(a)

    sigmas = [1.0, 5.0, 10.0]

    for order in [0,1,2]:
        for sigma in sigmas:
            res_ff = ff.gaussian2d(a, order, sigma)
            res_vigra = np.zeros_like(a)

            for c in range(avigra.shape[2]):
                res_vigra[:,:,c] = vigra.filters.gaussianDerivative(a[:,:,c], sigma, [order,order])

            print("gaussian ", order, sigma, np.max(np.abs(res_ff - res_vigra)))

            if not np.allclose(res_ff, res_vigra, atol=1e-6):
                raise Exception("FAIL: ", order, sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.gradmag2d(a, sigma)
        res_vigra = vigra.filters.gaussianGradientMagnitude(avigra, sigma, accumulate=False)
        print("gradmag2d ", sigma, np.max(np.abs(res_ff - res_vigra)))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            import IPython; IPython.embed()
            raise Exception("FAIL: gradmag2d ", sigma, np.max(np.abs(res_ff - res_vigra)))


    for sigma in sigmas:
        res_ff = ff.laplacian2d(a, sigma)
        res_vigra = vigra.filters.laplacianOfGaussian(avigra, sigma)
        print("laplacian2d ", sigma, np.max(np.abs(res_ff - res_vigra)))

        if not np.allclose(res_ff, res_vigra, atol=1e-6):
            raise Exception("FAIL: laplacian2d ", sigma, np.max(np.abs(res_ff - res_vigra)))

    '''
    for sigma in sigmas:
        for sigma2 in sigmas:
            res_ff = ff.st2d(a, sigma2, sigma)#
            import IPython; IPython.embed()
            res_vigra = vigra.filters.structureTensorEigenvalues(avigra, sigma, sigma2).reshape((-1,2)).swapaxes(0,1)
            print("ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))

            if not np.allclose(res_ff, res_vigra, atol=1e-6) or np.any(np.isnan(np.abs(res_ff - res_vigra))):
                raise Exception("FAIL: ST", sigma, sigma2, np.max(np.abs(res_ff - res_vigra)))
    '''
