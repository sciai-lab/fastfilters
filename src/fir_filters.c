// fastfilters
// Copyright (c) 2016 Sven Peter
// sven.peter@iwr.uni-heidelberg.de or mail@svenpeter.me
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "fastfilters.h"
#include "common.h"

bool DLL_PUBLIC fastfilters_fir_gaussian2d(const fastfilters_array2d_t *inarray, unsigned order, double sigma,
                                           fastfilters_array2d_t *outarray, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t kx = NULL;

    kx = fastfilters_kernel_fir_gaussian(order, sigma, opt_window_ratio(options));
    if (!kx)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, kx, kx, outarray, options);

out:
    if (kx)
        fastfilters_kernel_fir_free(kx);
    return result;
}

bool DLL_PUBLIC fastfilters_fir_hog2d(const fastfilters_array2d_t *inarray, double sigma, fastfilters_array2d_t *out_xx,
                                      fastfilters_array2d_t *out_xy, fastfilters_array2d_t *out_yy,
                                      const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t k_smooth = NULL;
    fastfilters_kernel_fir_t k_first = NULL;
    fastfilters_kernel_fir_t k_second = NULL;

    k_smooth = fastfilters_kernel_fir_gaussian(0, sigma, opt_window_ratio(options));
    if (!k_smooth)
        goto out;

    k_first = fastfilters_kernel_fir_gaussian(1, sigma, opt_window_ratio(options));
    if (!k_first)
        goto out;

    k_second = fastfilters_kernel_fir_gaussian(2, sigma, opt_window_ratio(options));
    if (!k_second)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, k_second, k_smooth, out_xx, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, k_smooth, k_second, out_yy, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, k_first, k_first, out_xy, options);
    if (!result)
        goto out;

out:
    if (k_smooth)
        fastfilters_kernel_fir_free(k_smooth);
    if (k_first)
        fastfilters_kernel_fir_free(k_first);
    if (k_second)
        fastfilters_kernel_fir_free(k_second);
    return result;
}

static bool fastfilters_fir_deriv2d_inner(const fastfilters_array2d_t *inarray, double sigma, unsigned order,
                                          fastfilters_array2d_t *out0, fastfilters_array2d_t *out1,
                                          const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t k_smooth = NULL;
    fastfilters_kernel_fir_t k_deriv = NULL;

    k_smooth = fastfilters_kernel_fir_gaussian(0, sigma, opt_window_ratio(options));
    if (!k_smooth)
        goto out;

    k_deriv = fastfilters_kernel_fir_gaussian(order, sigma, opt_window_ratio(options));
    if (!k_deriv)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, k_deriv, k_smooth, out0, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve2d(inarray, k_smooth, k_deriv, out1, options);
    if (!result)
        goto out;

out:
    if (k_smooth)
        fastfilters_kernel_fir_free(k_smooth);
    if (k_deriv)
        fastfilters_kernel_fir_free(k_deriv);
    return result;
}

static bool fastfilters_fir_deriv2d(const fastfilters_array2d_t *inarray, double sigma, unsigned order,
                                    fastfilters_array2d_t *outarray, bool do_sqrt, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_array2d_t *tmparray = NULL;

    tmparray = fastfilters_array2d_alloc(inarray->n_x, inarray->n_y, inarray->n_channels);
    if (!tmparray)
        goto out;

    result = fastfilters_fir_deriv2d_inner(inarray, sigma, order, tmparray, outarray, options);
    if (!result)
        goto out;

    if (do_sqrt)
        fastfilters_combine_addsqrt2d(outarray, tmparray, outarray);
    else
        fastfilters_combine_add2d(outarray, tmparray, outarray);

out:
    if (tmparray)
        fastfilters_array2d_free(tmparray);
    return result;
}

bool DLL_PUBLIC fastfilters_fir_gradmag2d(const fastfilters_array2d_t *inarray, double sigma,
                                          fastfilters_array2d_t *outarray, const fastfilters_options_t *options)
{
    return fastfilters_fir_deriv2d(inarray, sigma, 1, outarray, true, options);
}

bool DLL_PUBLIC fastfilters_fir_laplacian2d(const fastfilters_array2d_t *inarray, double sigma,
                                            fastfilters_array2d_t *outarray, const fastfilters_options_t *options)
{
    return fastfilters_fir_deriv2d(inarray, sigma, 2, outarray, false, options);
}

bool DLL_PUBLIC fastfilters_fir_structure_tensor2d(const fastfilters_array2d_t *inarray, double sigma_outer,
                                                   double sigma_inner, fastfilters_array2d_t *out_xx,
                                                   fastfilters_array2d_t *out_xy, fastfilters_array2d_t *out_yy,
                                                   const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_array2d_t *tmp = NULL;
    fastfilters_array2d_t *tmpx = NULL;
    fastfilters_array2d_t *tmpy = NULL;

    tmp = fastfilters_array2d_alloc(inarray->n_x, inarray->n_y, inarray->n_channels);
    if (!tmp)
        goto out;

    tmpx = fastfilters_array2d_alloc(inarray->n_x, inarray->n_y, inarray->n_channels);
    if (!tmpx)
        goto out;

    tmpy = fastfilters_array2d_alloc(inarray->n_x, inarray->n_y, inarray->n_channels);
    if (!tmpy)
        goto out;

    result = fastfilters_fir_deriv2d_inner(inarray, sigma_inner, 1, tmpx, tmpy, options);
    if (!result)
        goto out;

    fastfilters_combine_mul2d(tmpx, tmpx, tmp);
    result = fastfilters_fir_gaussian2d(tmp, 0, sigma_outer, out_xx, options);
    if (!result)
        goto out;

    fastfilters_combine_mul2d(tmpy, tmpy, tmp);
    result = fastfilters_fir_gaussian2d(tmp, 0, sigma_outer, out_yy, options);
    if (!result)
        goto out;

    fastfilters_combine_mul2d(tmpx, tmpy, tmp);
    result = fastfilters_fir_gaussian2d(tmp, 0, sigma_outer, out_xy, options);
    if (!result)
        goto out;

out:
    if (tmp)
        fastfilters_array2d_free(tmp);
    if (tmpx)
        fastfilters_array2d_free(tmpx);
    if (tmpy)
        fastfilters_array2d_free(tmpy);
    return result;
}

DLL_PUBLIC bool fastfilters_fir_hog3d(const fastfilters_array3d_t *inarray, double sigma, fastfilters_array3d_t *out_xx,
                                      fastfilters_array3d_t *out_yy, fastfilters_array3d_t *out_zz,
                                      fastfilters_array3d_t *out_xy, fastfilters_array3d_t *out_xz,
                                      fastfilters_array3d_t *out_yz, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t k_smooth = NULL;
    fastfilters_kernel_fir_t k_first = NULL;
    fastfilters_kernel_fir_t k_second = NULL;

    k_smooth = fastfilters_kernel_fir_gaussian(0, sigma, opt_window_ratio(options));
    if (!k_smooth)
        goto out;

    k_first = fastfilters_kernel_fir_gaussian(1, sigma, opt_window_ratio(options));
    if (!k_first)
        goto out;

    k_second = fastfilters_kernel_fir_gaussian(2, sigma, opt_window_ratio(options));
    if (!k_second)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_second, k_smooth, k_smooth, out_xx, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_smooth, k_second, k_smooth, out_yy, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_smooth, k_smooth, k_second, out_zz, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_first, k_first, k_smooth, out_xy, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_first, k_smooth, k_first, out_xz, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_smooth, k_first, k_first, out_yz, options);
    if (!result)
        goto out;

out:
    if (k_smooth)
        fastfilters_kernel_fir_free(k_smooth);
    if (k_first)
        fastfilters_kernel_fir_free(k_first);
    if (k_second)
        fastfilters_kernel_fir_free(k_second);
    return result;
}

bool DLL_PUBLIC fastfilters_fir_gaussian3d(const fastfilters_array3d_t *inarray, unsigned order, double sigma,
                                           fastfilters_array3d_t *outarray, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t kx = NULL;

    kx = fastfilters_kernel_fir_gaussian(order, sigma, opt_window_ratio(options));
    if (!kx)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, kx, kx, kx, outarray, options);

out:
    if (kx)
        fastfilters_kernel_fir_free(kx);
    return result;
}

static bool fastfilters_fir_deriv3d_inner(const fastfilters_array3d_t *inarray, double sigma, unsigned order,
                                          fastfilters_array3d_t *out0, fastfilters_array3d_t *out1,
                                          fastfilters_array3d_t *out2, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_kernel_fir_t k_smooth = NULL;
    fastfilters_kernel_fir_t k_deriv = NULL;

    k_smooth = fastfilters_kernel_fir_gaussian(0, sigma, opt_window_ratio(options));
    if (!k_smooth)
        goto out;

    k_deriv = fastfilters_kernel_fir_gaussian(order, sigma, opt_window_ratio(options));
    if (!k_deriv)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_deriv, k_smooth, k_smooth, out0, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_smooth, k_deriv, k_smooth, out1, options);
    if (!result)
        goto out;

    result = fastfilters_fir_convolve3d(inarray, k_smooth, k_smooth, k_deriv, out2, options);
    if (!result)
        goto out;

out:
    if (k_smooth)
        fastfilters_kernel_fir_free(k_smooth);
    if (k_deriv)
        fastfilters_kernel_fir_free(k_deriv);
    return result;
}

static bool fastfilters_fir_deriv3d(const fastfilters_array3d_t *inarray, double sigma, unsigned order,
                                    fastfilters_array3d_t *outarray, bool do_sqrt, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_array3d_t *tmparray0 = NULL;
    fastfilters_array3d_t *tmparray1 = NULL;

    tmparray0 = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmparray0)
        goto out;

    tmparray1 = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmparray1)
        goto out;

    result = fastfilters_fir_deriv3d_inner(inarray, sigma, order, outarray, tmparray0, tmparray1, options);
    if (!result)
        goto out;

    if (do_sqrt)
        fastfilters_combine_addsqrt3d(outarray, tmparray0, tmparray1, outarray);
    else
        fastfilters_combine_add3d(outarray, tmparray0, tmparray1, outarray);

out:
    if (tmparray0)
        fastfilters_array3d_free(tmparray0);
    if (tmparray1)
        fastfilters_array3d_free(tmparray1);
    return result;
}

bool DLL_PUBLIC fastfilters_fir_gradmag3d(const fastfilters_array3d_t *inarray, double sigma,
                                          fastfilters_array3d_t *outarray, const fastfilters_options_t *options)
{
    return fastfilters_fir_deriv3d(inarray, sigma, 1, outarray, true, options);
}

bool DLL_PUBLIC fastfilters_fir_laplacian3d(const fastfilters_array3d_t *inarray, double sigma,
                                            fastfilters_array3d_t *outarray, const fastfilters_options_t *options)
{
    return fastfilters_fir_deriv3d(inarray, sigma, 2, outarray, false, options);
}

bool DLL_PUBLIC fastfilters_fir_structure_tensor3d(const fastfilters_array3d_t *inarray, double sigma_outer,
                                                   double sigma_inner, fastfilters_array3d_t *out_xx,
                                                   fastfilters_array3d_t *out_yy, fastfilters_array3d_t *out_zz,
                                                   fastfilters_array3d_t *out_xy, fastfilters_array3d_t *out_xz,
                                                   fastfilters_array3d_t *out_yz, const fastfilters_options_t *options)
{
    bool result = false;
    fastfilters_array3d_t *tmpx = NULL;
    fastfilters_array3d_t *tmpy = NULL;
    fastfilters_array3d_t *tmpz = NULL;
    fastfilters_array3d_t *tmp = NULL;

    tmp = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmp)
        goto out;

    tmpx = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmpx)
        goto out;

    tmpy = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmpy)
        goto out;

    tmpz = fastfilters_array3d_alloc(inarray->n_x, inarray->n_y, inarray->n_z, inarray->n_channels);
    if (!tmpz)
        goto out;

    result = fastfilters_fir_deriv3d_inner(inarray, sigma_inner, 1, tmpx, tmpy, tmpz, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpx, tmpx, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_xx, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpy, tmpy, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_yy, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpz, tmpz, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_zz, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpx, tmpy, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_xy, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpx, tmpz, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_xz, options);
    if (!result)
        goto out;

    fastfilters_combine_mul3d(tmpy, tmpz, tmp);
    result = fastfilters_fir_gaussian3d(tmp, 0, sigma_outer, out_yz, options);
    if (!result)
        goto out;

out:
    if (tmp)
        fastfilters_array3d_free(tmp);
    if (tmpx)
        fastfilters_array3d_free(tmpx);
    if (tmpy)
        fastfilters_array3d_free(tmpy);
    if (tmpz)
        fastfilters_array3d_free(tmpz);
    return result;
}