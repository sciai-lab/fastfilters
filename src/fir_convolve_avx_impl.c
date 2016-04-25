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
#define kernel_addsub_ss(a, b) ((a) + (b))
#else
#define kernel_addsub_ps(a, b) _mm256_sub_ps((a), (b))
#define kernel_addsub_ss(a, b) ((a) - (b))
#endif

static bool
    BOOST_PP_CAT(fname(0, param_boundary_left, param_boundary_right, param_symm, param_avxfma, FF_KERNEL_LEN_FNAME),
                 _rgb)(const float *inptr, const float *in_border_left, const float *in_border_right, size_t n_pixels,
                       size_t pixel_stride, size_t n_outer, size_t outer_stride, float *outptr,
                       size_t outptr_outer_stride, size_t borderptr_outer_stride, const fastfilters_kernel_fir_t kernel)
{
    static const unsigned int lcmtbl[8] = {0, 8, 8, 24, 8, 40, 24, 56};
    static const unsigned int lcmtbl_v[8] = {0, 1, 1, 3, 1, 5, 3, 7};

    if (unlikely(pixel_stride >= 8))
        return false;

#ifndef FF_BOUNDARY_PTR_RIGHT
    (void)in_border_right;
#endif
#ifndef FF_BOUNDARY_PTR_LEFT
    (void)in_border_left;
#endif
#if !defined(FF_BOUNDARY_PTR_LEFT) && !defined(FF_BOUNDARY_PTR_RIGHT)
    (void)borderptr_outer_stride;
#endif

    for (unsigned int y = 0; y < n_outer; ++y) {
        // take next line of pixels
        float *cur_output = outptr + y * outptr_outer_stride;
        const float *cur_input = inptr + y * outer_stride;

        // left border
        unsigned int x = 0;

#if defined(FF_BOUNDARY_MIRROR_LEFT) || defined(FF_BOUNDARY_PTR_LEFT)
        for (unsigned int c = 0; c < pixel_stride; ++c) {
            cur_input = inptr + y * outer_stride + c;
            cur_output = outptr + y * outptr_outer_stride + c;

#ifdef FF_BOUNDARY_MIRROR_LEFT
            for (x = 0; x < FF_KERNEL_LEN; ++x) {
                float sum = kernel->coefs[0] * cur_input[x * pixel_stride];

                for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                    unsigned int offset_left;
                    if (-(int)k + (int)x < 0)
                        offset_left = -x + k;
                    else
                        offset_left = x - k;

                    sum += kernel->coefs[k] *
                           kernel_addsub_ss(cur_input[(x + k) * pixel_stride], cur_input[offset_left * pixel_stride]);
                }

                cur_output[x * pixel_stride] = sum;
            }
#endif

#ifdef FF_BOUNDARY_PTR_LEFT
            for (x = 0; x < FF_KERNEL_LEN; ++x) {
                float sum = kernel->coefs[0] * cur_input[x * pixel_stride];

                for (unsigned int k = 1; k < x; ++k) {
                    float left;
                    if (-(int)k + (int)x < 0)
                        left = in_border_left[y * borderptr_outer_stride +
                                              (FF_KERNEL_LEN - (int)k + (int)x) * pixel_stride];
                    else
                        left = cur_input[(x - k) * pixel_stride];
                    sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[(x + k) * pixel_stride], left);
                }

                cur_output[x * pixel_stride] = sum;
            }
#endif
        }
#endif

#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
        const unsigned int avx_end_single = (n_pixels) & ~7;
#else
        const unsigned int avx_end_single = (n_pixels - FF_KERNEL_LEN) & ~7;
#endif
        // valid area
        if (likely(avx_end_single > 32)) {
            // align to 8 pixel boundary
            const unsigned int x_align = (x + 7) & ~7;
            const unsigned int x_avx_start = x;
            for (unsigned int c = 0; c < pixel_stride; ++c) {
                cur_input = inptr + y * outer_stride + c;
                cur_output = outptr + y * outptr_outer_stride + c;
                for (x = x_avx_start; x < x_align; ++x) {
                    float sum = kernel->coefs[0] * cur_input[x * pixel_stride];

                    for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                        sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[(x + k) * pixel_stride],
                                                                   *(cur_input + (x - k) * pixel_stride));
                    }

                    cur_output[x] = sum;
                }
            }

            const unsigned int step = lcmtbl[pixel_stride];
            const unsigned int avx_end_step = avx_end_single - avx_end_single % step;

            cur_input = inptr + y * outer_stride;
            cur_output = outptr + y * outptr_outer_stride;
            for (; x < avx_end_step; x += step) {
                for (unsigned int subx = 0; subx < lcmtbl_v[pixel_stride]; ++subx) {
                    __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
                    __m256 sum = _mm256_mul_ps(kernel_val, _mm256_loadu_ps(cur_input + x * pixel_stride + subx * 8));

                    for (unsigned int k = 0; k <= kernel->len; ++k) {
                        kernel_val = _mm256_broadcast_ss(kernel->coefs + k);

                        __m256 pixels =
                            kernel_addsub_ps(_mm256_loadu_ps(cur_input + (x + k) * pixel_stride + subx * 8),
                                             _mm256_loadu_ps(cur_input + (x - k) * pixel_stride + subx * 8));
                        sum = _mm256_fmadd_ps(pixels, kernel_val, sum);
                    }

                    _mm256_storeu_ps(cur_output + x * pixel_stride + subx * 8, sum);
                }
            }
        }

// finish pixels until boundary
#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
        const size_t n_pixels_end = n_pixels;
#else
        const size_t n_pixels_end = n_pixels - FF_KERNEL_LEN;
#endif

        const unsigned int xstart_noavx = x;
        for (unsigned int c = 0; c < pixel_stride; ++c) {
            cur_input = inptr + y * outer_stride + c;
            cur_output = outptr + y * outptr_outer_stride + c;

            for (x = xstart_noavx; x < n_pixels_end; ++x) {
                float sum = cur_input[x * pixel_stride] * kernel->coefs[0];

                for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k)
                    sum += kernel->coefs[k] *
                           kernel_addsub_ss(cur_input[(x + k) * pixel_stride], *(cur_input + (x - k) * pixel_stride));

                cur_output[x * pixel_stride] = sum;
            }
        }

// right border
#if defined(FF_BOUNDARY_MIRROR_RIGHT) || defined(FF_BOUNDARY_PTR_RIGHT)
        const unsigned int xstart_border = x;

        for (unsigned int c = 0; c < pixel_stride; ++c) {
            cur_input = inptr + y * outer_stride + c;
            cur_output = outptr + y * outptr_outer_stride + c;

            for (x = xstart_border; x < n_pixels; ++x) {
                float sum = cur_input[x * pixel_stride] * kernel->coefs[0];

                for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                    float right;

                    if (x + k >= n_pixels)
#ifdef FF_BOUNDARY_MIRROR_RIGHT
                        right = cur_input[n_pixels - (((k + x) % n_pixels) - 2) * pixel_stride];
#else
                        right = in_border_right[y * borderptr_outer_stride + (((k + x) % n_pixels) * pixel_stride)];
#endif
                    else
                        right = cur_input[(x + k) * pixel_stride];

                    sum += kernel->coefs[k] * kernel_addsub_ss(right, *(cur_input + (x - k) * pixel_stride));
                }

                cur_output[x * pixel_stride] = sum;
            }
        }
#endif
    }

    return false;
}

bool DLL_LOCAL fname(0, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
#ifndef FF_BOUNDARY_PTR_RIGHT
    (void)in_border_right;
#endif
#ifndef FF_BOUNDARY_PTR_LEFT
    (void)in_border_left;
#endif
#if !defined(FF_BOUNDARY_PTR_LEFT) && !defined(FF_BOUNDARY_PTR_RIGHT)
    (void)borderptr_outer_stride;
#endif

#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
    const unsigned int avx_end = (n_pixels) & ~31;
    const unsigned int avx_end_single = (n_pixels) & ~7;
#else
    const unsigned int avx_end = (n_pixels - FF_KERNEL_LEN) & ~31;
    const unsigned int avx_end_single = (n_pixels - FF_KERNEL_LEN) & ~7;
#endif

    if (pixel_stride != 1)
        return BOOST_PP_CAT(
            fname(0, param_boundary_left, param_boundary_right, param_symm, param_avxfma, FF_KERNEL_LEN_FNAME),
            _rgb)(inptr, in_border_left, in_border_right, n_pixels, pixel_stride, n_outer, outer_stride, outptr,
                  outptr_outer_stride, borderptr_outer_stride, kernel);

    for (unsigned int y = 0; y < n_outer; ++y) {
        // take next line of pixels
        float *cur_output = outptr + y * outptr_outer_stride;
        const float *cur_input = inptr + y * outer_stride;

        // left border
        unsigned int x = 0;

#ifdef FF_BOUNDARY_MIRROR_LEFT
        for (x = 0; x < FF_KERNEL_LEN; ++x) {
            float sum = kernel->coefs[0] * cur_input[x];

            for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                unsigned int offset_left;
                if (-(int)k + (int)x < 0)
                    offset_left = -x + k;
                else
                    offset_left = x - k;

                sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[x + k], cur_input[offset_left]);
            }

            cur_output[x] = sum;
        }
#endif

#ifdef FF_BOUNDARY_PTR_LEFT
        for (x = 0; x < FF_KERNEL_LEN; ++x) {
            float sum = kernel->coefs[0] * cur_input[x];

            for (unsigned int k = 1; k < x; ++k) {
                float left;
                if (-(int)k + (int)x < 0)
                    left = in_border_left[y * borderptr_outer_stride + (FF_KERNEL_LEN - (int)k + (int)x)];
                else
                    left = cur_input[x - k];

                sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[x + k], left);
            }

            cur_output[x] = sum;
        }
#endif

        if (likely(avx_end_single > 32)) {
            // align to 8 pixel boundary
            const unsigned int x_align = (x + 7) & ~7;
            for (; x < x_align; ++x) {
                float sum = kernel->coefs[0] * cur_input[x];

                for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                    sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[x + k], *(cur_input + x - k));
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

                    // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] =
                    // (image[i-j] +
                    // image[i+j]) * kernel[j])
                    // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                    __m256 pixels0, pixels1, pixels2, pixels3;

                    pixels0 =
                        kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j), _mm256_loadu_ps(cur_input + (x - j)));
                    pixels1 = kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 8),
                                               _mm256_loadu_ps(cur_input + (x - j) + 8));
                    pixels2 = kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 16),
                                               _mm256_loadu_ps(cur_input + (x - j) + 16));
                    pixels3 = kernel_addsub_ps(_mm256_loadu_ps(cur_input + x + j + 24),
                                               _mm256_loadu_ps(cur_input + (x - j) + 24));

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
        }
// finish pixels until boundary
#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
        const size_t n_pixels_end = n_pixels;
#else
        const size_t n_pixels_end = n_pixels - FF_KERNEL_LEN;
#endif
        for (; x < n_pixels_end; ++x) {
            float sum = cur_input[x] * kernel->coefs[0];

            for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                sum += kernel->coefs[k] * kernel_addsub_ss(cur_input[x + k], *(cur_input + x - k));
            }

            cur_output[x] = sum;
        }

// right border
#if defined(FF_BOUNDARY_MIRROR_RIGHT) || defined(FF_BOUNDARY_PTR_RIGHT)
        for (; x < n_pixels; ++x) {
            float sum = cur_input[x] * kernel->coefs[0];

            for (unsigned int k = 1; k <= FF_KERNEL_LEN; ++k) {
                float right;

                if (x + k >= n_pixels)
#ifdef FF_BOUNDARY_MIRROR_RIGHT
                    right = cur_input[n_pixels - ((k + x) % n_pixels) - 2];
#else
                    right = in_border_right[y * borderptr_outer_stride + ((k + x) % n_pixels)];
#endif
                else
                    right = cur_input[x + k];

                sum += kernel->coefs[k] * kernel_addsub_ss(right, *(cur_input + x - k));
            }

            cur_output[x] = sum;
        }
#endif
    }

    return true;
}

bool DLL_LOCAL fname(1, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
#ifndef FF_BOUNDARY_PTR_RIGHT
    (void)in_border_right;
#endif
#ifndef FF_BOUNDARY_PTR_LEFT
    (void)in_border_left;
#endif
#if !defined(FF_BOUNDARY_PTR_LEFT) && !defined(FF_BOUNDARY_PTR_RIGHT)
    (void)borderptr_outer_stride;
#endif

    if (unlikely(outer_stride != 1))
        return false;

    const unsigned int avx_end = n_outer & ~7;
    const unsigned int noavx_left = n_outer - avx_end;
    const unsigned int n_outer_aligned = (n_outer + 8) & ~7;

    const __m256i mask =
        _mm256_set_epi32(0, noavx_left >= 7 ? 0xffffffff : 0, noavx_left >= 6 ? 0xffffffff : 0,
                         noavx_left >= 5 ? 0xffffffff : 0, noavx_left >= 4 ? 0xffffffff : 0,
                         noavx_left >= 3 ? 0xffffffff : 0, noavx_left >= 2 ? 0xffffffff : 0, 0xffffffff);

    float *tmp = fastfilters_memory_align(32, (FF_KERNEL_LEN + 1) * n_outer_aligned * sizeof(float));

    if (!tmp)
        return false;

    size_t pixel = 0;

// left border
#if defined(FF_BOUNDARY_MIRROR_LEFT) || defined(FF_BOUNDARY_PTR_LEFT)
    for (; pixel < FF_KERNEL_LEN; ++pixel) {
        const float *cur_inptr = inptr + pixel * pixel_stride;
        const unsigned tmpidx = pixel % (FF_KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer_aligned;

        unsigned dim;
        for (dim = 0; dim < avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(cur_inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);
                __m256 pixel_left;

                if (i > pixel) {
#ifdef FF_BOUNDARY_MIRROR_LEFT
                    pixel_left = _mm256_loadu_ps(inptr + (i - pixel) * pixel_stride + dim);
#else
                    pixel_left = _mm256_loadu_ps(in_border_left +
                                                 (FF_KERNEL_LEN + (int)(pixel - i)) * borderptr_outer_stride + dim);
#endif
                } else
                    pixel_left = _mm256_loadu_ps(inptr + (pixel - i) * pixel_stride + dim);

                pixels = kernel_addsub_ps(_mm256_loadu_ps(inptr + (pixel + i) * pixel_stride + dim), pixel_left);
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (noavx_left > 0) {
            __m256 pixels = _mm256_maskload_ps(cur_inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);
                __m256 pixel_left;

                if (i > pixel) {
#ifdef FF_BOUNDARY_MIRROR_LEFT
                    pixel_left = _mm256_maskload_ps(inptr + (i - pixel) * pixel_stride + dim, mask);
#else
                    pixel_left = _mm256_maskload_ps(
                        in_border_left + (FF_KERNEL_LEN + (int)(pixel - i)) * borderptr_outer_stride + dim, mask);
#endif
                } else
                    pixel_left = _mm256_maskload_ps(inptr + (pixel - i) * pixel_stride + dim, mask);

                pixels =
                    kernel_addsub_ps(_mm256_maskload_ps(inptr + (pixel + i) * pixel_stride + dim, mask), pixel_left);
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }
    }
#endif

// valid part of line
#if defined(FF_BOUNDARY_OPTIMISTIC_RIGHT)
    const size_t pixel_end = n_pixels;
#else
    const size_t pixel_end = n_pixels - FF_KERNEL_LEN;
#endif

    for (; pixel < pixel_end; ++pixel) {
        const float *cur_inptr = inptr + pixel * pixel_stride;
        const unsigned tmpidx = pixel % (FF_KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer_aligned;

        unsigned dim;
        for (dim = 0; dim < avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(cur_inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);

                pixels = kernel_addsub_ps(_mm256_loadu_ps(inptr + (pixel + i) * pixel_stride + dim),
                                          _mm256_loadu_ps(inptr + (pixel - i) * pixel_stride + dim));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (noavx_left > 0) {
            __m256 pixels = _mm256_maskload_ps(cur_inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);

                pixels = kernel_addsub_ps(_mm256_maskload_ps(inptr + (pixel + i) * pixel_stride + dim, mask),
                                          _mm256_maskload_ps(inptr + (pixel - i) * pixel_stride + dim, mask));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        const unsigned writeidx = (pixel + 1) % (FF_KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer_aligned;
        memcpy(outptr + (pixel - FF_KERNEL_LEN) * outptr_outer_stride, writeptr, n_outer * sizeof(float));
    }

// right border
#if defined(FF_BOUNDARY_PTR_RIGHT) || defined(FF_BOUNDARY_MIRROR_RIGHT)
    // right border
    for (; pixel < n_pixels; ++pixel) {
        const float *cur_inptr = inptr + pixel * pixel_stride;
        const unsigned tmpidx = pixel % (FF_KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer_aligned;

        unsigned dim;
        for (dim = 0; dim < avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(cur_inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);
                __m256 pixel_right;

                if (pixel + i < n_pixels)
                    pixel_right = _mm256_loadu_ps(inptr + (pixel + i) * pixel_stride + dim);
                else {
#ifdef FF_BOUNDARY_PTR_RIGHT
                    pixel_right =
                        _mm256_loadu_ps(in_border_right + ((i + pixel) % n_pixels) * borderptr_outer_stride + dim);
#endif
#ifdef FF_BOUNDARY_MIRROR_RIGHT
                    pixel_right =
                        _mm256_loadu_ps(inptr + (n_pixels - ((i + pixel) % n_pixels) - 2) * pixel_stride + dim);
#endif
                }

                pixels = kernel_addsub_ps(pixel_right, _mm256_loadu_ps(inptr + (pixel - i) * pixel_stride + dim));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (noavx_left > 0) {
            __m256 pixels = _mm256_maskload_ps(inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(kernel->coefs);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= FF_KERNEL_LEN; ++i) {
                kernel_val = _mm256_broadcast_ss(kernel->coefs + i);
                __m256 pixel_right;

                if (pixel + i < n_pixels)
                    pixel_right = _mm256_maskload_ps(inptr + (pixel + i) * pixel_stride + dim, mask);
                else {
#ifdef FF_BOUNDARY_PTR_RIGHT
                    pixel_right = _mm256_maskload_ps(
                        in_border_right + ((i + pixel) % n_pixels) * borderptr_outer_stride + dim, mask);
#endif
#ifdef FF_BOUNDARY_MIRROR_RIGHT
                    pixel_right = _mm256_maskload_ps(
                        inptr + (n_pixels - ((i + pixel) % n_pixels) - 2) * pixel_stride + dim, mask);
#endif
                }

                pixels =
                    kernel_addsub_ps(pixel_right, _mm256_maskload_ps(inptr + (pixel - i) * pixel_stride + dim, mask));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        const unsigned writeidx = (pixel + 1) % (FF_KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer_aligned;
        memcpy(outptr + (pixel - FF_KERNEL_LEN) * outptr_outer_stride, writeptr, n_outer * sizeof(float));
    }
#endif

    // copy from scratch memory to real output
    for (unsigned i = 0; i < FF_KERNEL_LEN; ++i) {
        unsigned pixel = n_pixels + i;
        const unsigned writeidx = (pixel + 1) % (FF_KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer_aligned;
        memcpy(outptr + (pixel - FF_KERNEL_LEN) * outptr_outer_stride, writeptr, n_outer * sizeof(float));
    }

    fastfilters_memory_align_free(tmp);

    return true;
}

#undef param_symm
#undef param_boundary_left
#undef param_boundary_right
#undef FF_KERNEL_LEN_FNAME
#undef kernel_addsub_ps
#undef kernel_addsub_ss

#endif
