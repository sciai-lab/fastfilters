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

#include "fastfilters.h"
#include "common.h"
#include "config.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>

#ifndef __FMA__
#define _mm256_fmadd_ps(a, b, c) (_mm256_add_ps(_mm256_mul_ps((a), (b)), (c)))
#endif

#include "fir_convolve_avx_common.h"

#if !defined(FF_KERNEL_LEN) && !defined(FF_KERNEL_LEN_RUNTIME)
#error !defined(FF_KERNEL_LEN) && !defined(FF_KERNEL_LEN_RUNTIME)
#endif

#if !defined(FF_BOUNDARY_OPTIMISTIC_LEFT) && !defined(FF_BOUNDARY_MIRROR_LEFT) && !defined(FF_BOUNDARY_PTR_LEFT)

#define FF_BOUNDARY_OPTIMISTIC_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_OPTIMISTIC_LEFT

#define FF_BOUNDARY_MIRROR_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_MIRROR_LEFT

#define FF_BOUNDARY_PTR_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_PTR_LEFT

#elif !defined(FF_BOUNDARY_OPTIMISTIC_RIGHT) && !defined(FF_BOUNDARY_MIRROR_RIGHT) && !defined(FF_BOUNDARY_PTR_RIGHT)

#define FF_BOUNDARY_OPTIMISTIC_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_OPTIMISTIC_RIGHT

#define FF_BOUNDARY_MIRROR_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_MIRROR_RIGHT

#define FF_BOUNDARY_PTR_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_PTR_RIGHT

#elif !defined(FF_KERNEL_SYMMETRIC) && !defined(FF_KERNEL_ANTISYMMETRIC)

#define FF_KERNEL_SYMMETRIC
#include "fir_convolve_avx_impl.c"
#undef FF_KERNEL_SYMMETRIC

#define FF_KERNEL_ANTISYMMETRIC
#include "fir_convolve_avx_impl.c"
#undef FF_KERNEL_ANTISYMMETRIC

#else

#ifdef FF_BOUNDARY_MIRROR_LEFT
#define param_boundary_left 0
#elif defined(FF_BOUNDARY_OPTIMISTIC_LEFT)
#define param_boundary_left 1
#elif defined(FF_BOUNDARY_PTR_LEFT)
#define param_boundary_left 2
#else
#error "Unknown border left"
#endif

#ifdef FF_BOUNDARY_MIRROR_RIGHT
#define param_boundary_right 0
#elif defined(FF_BOUNDARY_OPTIMISTIC_RIGHT)
#define param_boundary_right 1
#elif defined(FF_BOUNDARY_PTR_RIGHT)
#define param_boundary_right 2
#else
#error "Unknown border right"
#endif

#ifdef FF_KERNEL_SYMMETRIC
#define param_symm 1
#elif defined(FF_KERNEL_ANTISYMMETRIC)
#define param_symm 0
#else
#error "Unknown symmetry"
#endif

#ifdef FF_KERNEL_LEN_RUNTIME
#define FF_KERNEL_LEN kernel->len
#define FF_KERNEL_LEN_FNAME N
#else
#define FF_KERNEL_LEN_FNAME FF_KERNEL_LEN
#endif

#ifdef FF_KERNEL_SYMMETRIC
#define kernel_addsub_ps(a, b) _mm256_add_ps((a), (b))
#else
#define kernel_addsub_ps(a, b) _mm256_sub_ps((a), (b))
#endif

bool DLL_LOCAL fname(0, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
    const unsigned int avx_end = (n_pixels) & ~31;
    const unsigned int avx_end_single = (n_pixels) & ~7;
#else
    const unsigned int avx_end = (n_pixels - FF_KERNEL_LEN) & ~31;
    const unsigned int avx_end_single = (n_pixels - FF_KERNEL_LEN) & ~7;
#endif

    for (unsigned int y = 0; y < n_outer; ++y) {
        // take next line of pixels
        float *cur_output = outptr + y * outptr_outer_stride;
        const float *cur_input = inptr + y * outer_stride;

        // left border
        unsigned int x = 0;

#ifdef FF_BOUNDARY_MIRROR_LEFT
        for (x = 0; x < FF_KERNEL_LEN; ++x) {
            float sum = kernel->coefs[0] * cur_input[x];

            for (unsigned int k = 1; k < x; ++k) {
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_input[x + k] + cur_input[x - k]);
#else
                sum += kernel->coefs[k] * (cur_input[x + k] - cur_input[x - k]);
#endif
            }

            for (unsigned int k = x; k <= FF_KERNEL_LEN; ++k) {
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_input[x + k] + cur_input[k - x]);
#else
                sum += kernel->coefs[k] * (cur_input[x + k] - cur_input[k - x]);
#endif
            }

            cur_output[x] = sum;
        }
#endif
#ifdef FF_BOUNDARY_PTR_LEFT
#endif

        // align to 8 pixel boundary
        const unsigned int x_align = (x + 7) & ~7;
        for (; x < x_align; ++x) {
            float sum = kernel->coefs[0] * cur_input[x];

            for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_input[x + k] + cur_input[x - k]);
#else
                sum += kernel->coefs[k] * (cur_input[x + k] - cur_input[x - k]);
#endif
            }

            cur_output[x] = sum;
        }

        // align to 32 pixel boundary
        for (; x < 32; x += 8) {
            __m256 result = _mm256_loadu_ps(cur_input + x);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel->coefs[0]);

            result = _mm256_mul_ps(result, kernel_val);

            for (unsigned j = 1; j <= FF_KERNEL_LEN; ++j) {
                __m256 pixels;
                kernel_val = _mm256_broadcast_ss(&kernel->coefs[j]);

                pixels = kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j), _mm256_loadu_ps(cur_input + x - j));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_storeu_ps(cur_output + x, result);
        }

        // main loop - 32 pixels at once
        for (; x < avx_end; x += 32) {
            // load next 32 pixels
            __m256 result0 = _mm256_loadu_ps(cur_input + x);
            __m256 result1 = _mm256_loadu_ps(cur_input + x + 8);
            __m256 result2 = _mm256_loadu_ps(cur_input + x + 16);
            __m256 result3 = _mm256_loadu_ps(cur_input + x + 24);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_broadcast_ss(&kernel->coefs[0]);
            result0 = _mm256_mul_ps(result0, kernel_val);
            result1 = _mm256_mul_ps(result1, kernel_val);
            result2 = _mm256_mul_ps(result2, kernel_val);
            result3 = _mm256_mul_ps(result3, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (unsigned int j = 1; j <= FF_KERNEL_LEN; ++j) {
                kernel_val = _mm256_broadcast_ss(&kernel->coefs[j]);

                // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] = (image[i-j] +
                // image[i+j]) * kernel[j])
                // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                __m256 pixels0, pixels1, pixels2, pixels3;

                pixels0 = kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j), _mm256_loadu_ps(cur_input + x - j));
                pixels1 =
                    kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 8), _mm256_loadu_ps(cur_input + x - j + 8));
                pixels2 =
                    kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 16), _mm256_loadu_ps(cur_input + x - j + 16));
                pixels3 =
                    kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 24), _mm256_loadu_ps(cur_input + x - j + 24));

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
                result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
                result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
                result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
            }

            _mm256_storeu_ps(cur_output + x, result0);
            _mm256_storeu_ps(cur_output + x + 8, result1);
            _mm256_storeu_ps(cur_output + x + 16, result2);
            _mm256_storeu_ps(cur_output + x + 24, result3);
        }

        // align until we have to switch to non-SIMD
        for (; x < avx_end_single; x += 8) {
            __m256 result = _mm256_loadu_ps(cur_input + x);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel->coefs[0]);

            result = _mm256_mul_ps(result, kernel_val);

            for (unsigned j = 1; j <= FF_KERNEL_LEN; ++j) {
                kernel_val = _mm256_broadcast_ss(&kernel->coefs[j]);
                __m256 pixels =
                    kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j), _mm256_loadu_ps(cur_input + x - j));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_storeu_ps(cur_output + x, result);
        }

// finish pixels until boundary
#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
        const size_t n_pixels_end = n_pixels;
#else
        const size_t n_pixels_end = n_pixels - FF_KERNEL_LEN - 1;
#endif
        for (; x < n_pixels_end; ++x) {
            float sum = cur_input[x] * kernel->coefs[0];

            for (unsigned int k = 0; k <= FF_KERNEL_LEN; ++k) {
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_input[x + k] + cur_input[x - k]);
#else
                sum += kernel->coefs[k] * (cur_input[x + k] - cur_input[x - k]);
#endif
            }

            cur_output[x] = sum;
        }

// right border
#ifdef FF_BOUNDARY_MIRROR_RIGHT
#endif
#ifdef FF_BOUNDARY_PTR_RIGHT
#endif
    }

    (void)avx_end;
    (void)avx_end_single;

    (void)inptr;
    (void)in_border_left;
    (void)in_border_right;
    (void)n_pixels;
    (void)pixel_stride;
    (void)n_outer;
    (void)outer_stride;
    (void)outptr;
    (void)outptr_outer_stride;
    (void)borderptr_outer_stride;
    (void)kernel;
    return false;
}

bool DLL_LOCAL fname(1, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
    (void)inptr;
    (void)in_border_left;
    (void)in_border_right;
    (void)n_pixels;
    (void)pixel_stride;
    (void)n_outer;
    (void)outer_stride;
    (void)outptr;
    (void)outptr_outer_stride;
    (void)borderptr_outer_stride;
    (void)kernel;
    return false;
}

#undef param_symm
#undef param_boundary_left
#undef param_boundary_right
#undef FF_KERNEL_LEN_FNAME
#undef kernel_addsub_ps

#endif
