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

#include <boost/preprocessor/list/cat.hpp>

#if !defined(FF_BOUNDARY_OPTIMISTIC) && !defined(FF_BOUNDARY_MIRROR)

#define FF_BOUNDARY_OPTIMISTIC
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_OPTIMISTIC

#define FF_BOUNDARY_MIRROR
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_MIRROR

#elif !defined(FF_KERNEL_SYMMETRIC) && !defined(FF_KERNEL_ANTISYMMETRIC)

#define FF_KERNEL_SYMMETRIC
#include "fir_convolve_nosimd_impl.h"
#undef FF_KERNEL_SYMMETRIC

#define FF_KERNEL_ANTISYMMETRIC
#include "fir_convolve_nosimd_impl.h"
#undef FF_KERNEL_ANTISYMMETRIC

#else

#ifdef FF_BOUNDARY_OPTIMISTIC
#define boundary_name optimistic_
#elif defined(FF_BOUNDARY_MIRROR)
#define boundary_name mirror_
#else
#error "No boundary treatment mode defined."
#endif

#ifdef FF_KERNEL_SYMMETRIC
#define symmetry_name symmetric
#elif defined(FF_KERNEL_ANTISYMMETRIC)
#define symmetry_name antisymmetric
#else
#error "FF_KERNEL_SYMMETRIC and FF_KERNEL_ANTISYMMETRIC not defined"
#endif

#define KERNEL_LEN BOOST_PP_ITERATION()

// hack to make waf dependency tracking + warning generation work correctly
// this files in included to create a function called fir_convolve_impl_optimistic_symmetric0 that is never used
// in the header of fir_convolve_nosimd.c
// mapping the kernel length to 10 here avoid a spurious warning about an unsigned < 0 compare
#if KERNEL_LEN == 0
#undef KERNEL_LEN
#define KERNEL_LEN 10
#endif

static bool BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_impl_, BOOST_PP_CAT(boundary_name, symmetry_name)),
                         BOOST_PP_ITERATION())(const float *inptr, size_t n_pixels, size_t pixel_stride, size_t n_outer,
                                               size_t outer_stride, float *outptr, const float *kernel)
{
    for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
        const float *cur_inptr = inptr + outer_stride * i_outer;
        float *cur_outptr = outptr + outer_stride * i_outer;

        unsigned int i_inner = 0;

// left border
#ifdef FF_BOUNDARY_MIRROR
        for (unsigned int j = 0; j < KERNEL_LEN; ++j) {
            float sum = 0.0;

            sum = kernel[0] * cur_inptr[0];

            for (unsigned int k = 1; k < KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)j < 0)
                    offset_left = -j + k;
                else
                    offset_left = j - k;

                if (k + j >= n_pixels)
                    offset_right = n_pixels - ((k + j) % n_pixels) - 2;
                else
                    offset_right = j + k;
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel[k] * (cur_inptr[offset_right * pixel_stride] + cur_inptr[offset_left * pixel_stride]);
#else
                sum += kernel[k] * (cur_inptr[offset_right * pixel_stride] - cur_inptr[offset_left * pixel_stride]);
#endif
            }

            cur_outptr[j * pixel_stride] = sum;
        }

        i_inner = KERNEL_LEN;
#endif

// 'valid' area of line
#ifdef FF_BOUNDARY_MIRROR
        const unsigned int end = n_pixels - KERNEL_LEN;
#else
        const unsigned int end = n_pixels;
#endif
        for (; i_inner < end; ++i_inner) {
            float sum = kernel[0] * cur_inptr[0];

            for (unsigned int k = 1; k < KERNEL_LEN; ++k) {
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel[k] * (cur_inptr[(i_inner + k) * pixel_stride] + cur_inptr[(i_inner - k) * pixel_stride]);
#else
                sum += kernel[k] * (cur_inptr[(i_inner + k) * pixel_stride] - cur_inptr[(i_inner - k) * pixel_stride]);
#endif
            }

            cur_outptr[i_inner * pixel_stride] = sum;
        }

// right border
#ifdef FF_BOUNDARY_MIRROR
        for (; i_inner < n_pixels; ++i_inner) {
            float sum = 0.0;

            sum = kernel[0] * cur_inptr[0];

            for (unsigned int k = 1; k < KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)i_inner < 0)
                    offset_left = -i_inner + k;
                else
                    offset_left = i_inner - k;

                if (k + i_inner >= n_pixels)
                    offset_right = n_pixels - ((k + i_inner) % n_pixels) - 2;
                else
                    offset_right = i_inner + k;
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel[k] * (cur_inptr[offset_right * pixel_stride] + cur_inptr[offset_left * pixel_stride]);
#else
                sum += kernel[k] * (cur_inptr[offset_right * pixel_stride] - cur_inptr[offset_left * pixel_stride]);
#endif
            }

            cur_outptr[i_inner * pixel_stride] = sum;
        }

        i_inner = KERNEL_LEN;
#endif
    }

    return true;
}

#undef symmetry_name
#undef boundary_name
#undef KERNEL_LEN
#endif