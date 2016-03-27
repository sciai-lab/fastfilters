#include "fastfilters.hxx"
#include "util.hxx"

#include <immintrin.h>
#include <stdlib.h>
#include <type_traits>
#include <stdexcept>
#include <iostream>
#include <string.h>

namespace fastfilters
{

namespace fir
{

template <bool is_symmetric>
static void internal_convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels,
                                                   const unsigned n_times, const unsigned dim_stride, float *output,
                                                   Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();
    const unsigned int half_kernel_len = kernel.half_len();
    const unsigned int avx_end = (n_pixels - kernel_len) & ~31;
    const unsigned int avx_end_single = (n_pixels - kernel_len) & ~7;

    float *tmp = (float *)detail::avx_memalign(n_pixels*sizeof(float));

    for (unsigned int dim = 0; dim < n_times; ++dim) {
        // take next line of pixels
        float *cur_output = output + dim * dim_stride;

        unsigned int j;
        for (j = 0; j < (n_pixels & ~7); j += 8)
            _mm256_store_ps(tmp + j, _mm256_loadu_ps(input + dim * dim_stride + j));
        for (; j < n_pixels; ++j)
            tmp[j] = input[dim * dim_stride + j];

        // this function is only used for small kernels (<25 pixel)
        // such that non-vectorized code can be used for the border
        // treament without speed penalties
        unsigned int i = 0;
        for (i = 0; i < half_kernel_len; ++i) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + (int)i < 0)
                    offset = -i - kreal;
                else
                    offset = i + kreal;
                sum += kernel[k] * tmp[offset];
            }

            cur_output[i] = sum;
        }

        // working on 32 pixels at the same time leads to a fully used ymm register bank
        for (; i < avx_end; i += 32) {
            // load next 32 pixels
            __m256 result0 = _mm256_load_ps(tmp + i);
            __m256 result1 = _mm256_load_ps(tmp + i + 8);
            __m256 result2 = _mm256_load_ps(tmp + i + 16);
            __m256 result3 = _mm256_load_ps(tmp + i + 24);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            result0 = _mm256_mul_ps(result0, kernel_val);
            result1 = _mm256_mul_ps(result1, kernel_val);
            result2 = _mm256_mul_ps(result2, kernel_val);
            result3 = _mm256_mul_ps(result3, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[j]);

                // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] = (image[i-j] +
                // image[i+j]) * kernel[j])
                // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                __m256 pixels0, pixels1, pixels2, pixels3;

                if (is_symmetric) {
                    pixels0 = _mm256_add_ps(_mm256_loadu_ps(tmp + i + j), _mm256_loadu_ps(tmp + i - j));
                    pixels1 = _mm256_add_ps(_mm256_loadu_ps(tmp + i + j + 8), _mm256_loadu_ps(tmp + i - j + 8));
                    pixels2 = _mm256_add_ps(_mm256_loadu_ps(tmp + i + j + 16), _mm256_loadu_ps(tmp + i - j + 16));
                    pixels3 = _mm256_add_ps(_mm256_loadu_ps(tmp + i + j + 24), _mm256_loadu_ps(tmp + i - j + 24));
                } else {
                    pixels0 = _mm256_sub_ps(_mm256_loadu_ps(tmp + i + j), _mm256_loadu_ps(tmp + i - j));
                    pixels1 = _mm256_sub_ps(_mm256_loadu_ps(tmp + i + j + 8), _mm256_loadu_ps(tmp + i - j + 8));
                    pixels2 = _mm256_sub_ps(_mm256_loadu_ps(tmp + i + j + 16), _mm256_loadu_ps(tmp + i - j + 16));
                    pixels3 = _mm256_sub_ps(_mm256_loadu_ps(tmp + i + j + 24), _mm256_loadu_ps(tmp + i - j + 24));
                }

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
                result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
                result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
                result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
            }

            // write result to output array
            _mm256_storeu_ps(cur_output + i, result0);
            _mm256_storeu_ps(cur_output + i + 8, result1);
            _mm256_storeu_ps(cur_output + i + 16, result2);
            _mm256_storeu_ps(cur_output + i + 24, result3);
        }

        // fast path for up to 24 pixels - only results in measureable speedup for small lines
        // where this is actually the bottleneck
        for (; i < avx_end_single; i += 8) {
            __m256 result = _mm256_load_ps(tmp + i);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            ;
            result = _mm256_mul_ps(result, kernel_val);

            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[j]);
                __m256 pixels;

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_loadu_ps(tmp + i + j), _mm256_loadu_ps(tmp + i - j));
                else
                    pixels = _mm256_sub_ps(_mm256_loadu_ps(tmp + i + j), _mm256_loadu_ps(tmp + i - j));

                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_storeu_ps(cur_output + i, result);
        }

        // right border
        for (; i < n_pixels; ++i) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + i >= n_pixels)
                    offset = n_pixels - ((kreal + i) % n_pixels) - 2;
                else
                    offset = i + kreal;
                sum += kernel[k] * tmp[offset];
            }

            cur_output[i] = sum;
        }
    }

    detail::avx_free(tmp);
}

void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   const unsigned dim_stride, float *output, Kernel &kernel)
{
    if (kernel.is_symmetric)
        internal_convolve_fir_inner_single_avx<true>(input, n_pixels, n_times, dim_stride, output, kernel);
    else
        internal_convolve_fir_inner_single_avx<false>(input, n_pixels, n_times, dim_stride, output, kernel);
}

template <bool is_symmetric>
static void internal_convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels,
                                                   const unsigned int pixel_stride, const unsigned n_times,
                                                   float *output, Kernel &kernel)
{
    const unsigned int half_kernel_len = kernel.half_len();
    const unsigned int dim_avx_end = n_times & ~7;
    const unsigned int dim_left = n_times - dim_avx_end;
    const unsigned int n_dims_aligned = (n_times + 8) & ~7;

    const __m256i mask = _mm256_set_epi32(0, dim_left >= 7 ? 0xffffffff : 0, dim_left >= 6 ? 0xffffffff : 0,
                                          dim_left >= 5 ? 0xffffffff : 0, dim_left >= 4 ? 0xffffffff : 0,
                                          dim_left >= 3 ? 0xffffffff : 0, dim_left >= 2 ? 0xffffffff : 0, 0xffffffff);

    float *test = (float *)detail::avx_memalign(n_dims_aligned * sizeof(float) * (half_kernel_len + 1));

    // left border
    for (unsigned pixel = 0; pixel < half_kernel_len; ++pixel) {
        const float *inptr = input + pixel * pixel_stride;
        float *tmpptr = test + pixel * n_dims_aligned;

        unsigned dim;
        for (dim = 0; dim < dim_avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);
                __m256 pixel_mirrored;

                if (i > pixel)
                    pixel_mirrored = _mm256_loadu_ps(input + (i - pixel) * pixel_stride + dim);
                else
                    pixel_mirrored = _mm256_loadu_ps(input + (pixel - i) * pixel_stride + dim);

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_loadu_ps(input + (pixel + i) * pixel_stride + dim), pixel_mirrored);
                else
                    pixels = _mm256_sub_ps(_mm256_loadu_ps(input + (pixel + i) * pixel_stride + dim), pixel_mirrored);
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (dim_left > 0) {
            __m256 pixels = _mm256_maskload_ps(inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);
                __m256 pixel_mirrored;

                if (i > pixel)
                    pixel_mirrored = _mm256_maskload_ps(input + (i - pixel) * pixel_stride + dim, mask);
                else
                    pixel_mirrored = _mm256_maskload_ps(input + (pixel - i) * pixel_stride + dim, mask);

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_maskload_ps(input + (pixel + i) * pixel_stride + dim, mask),
                                           pixel_mirrored);
                else
                    pixels = _mm256_sub_ps(_mm256_maskload_ps(input + (pixel + i) * pixel_stride + dim, mask),
                                           pixel_mirrored);
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }
    }

    for (unsigned pixel = half_kernel_len; pixel < n_pixels - half_kernel_len; ++pixel) {
        const float *inptr = input + pixel * pixel_stride;
        const unsigned tmpidx = pixel % (half_kernel_len + 1);
        float *tmpptr = test + tmpidx * n_dims_aligned;

        unsigned dim;
        for (dim = 0; dim < dim_avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_loadu_ps(input + (pixel + i) * pixel_stride + dim),
                                           _mm256_loadu_ps(input + (pixel - i) * pixel_stride + dim));
                else
                    pixels = _mm256_sub_ps(_mm256_loadu_ps(input + (pixel + i) * pixel_stride + dim),
                                           _mm256_loadu_ps(input + (pixel - i) * pixel_stride + dim));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (dim_left > 0) {
            __m256 pixels = _mm256_maskload_ps(inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_maskload_ps(input + (pixel + i) * pixel_stride + dim, mask),
                                           _mm256_maskload_ps(input + (pixel - i) * pixel_stride + dim, mask));
                else
                    pixels = _mm256_sub_ps(_mm256_maskload_ps(input + (pixel + i) * pixel_stride + dim, mask),
                                           _mm256_maskload_ps(input + (pixel - i) * pixel_stride + dim, mask));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        const unsigned writeidx = (pixel + 1) % (half_kernel_len + 1);
        float *writeptr = test + writeidx * n_dims_aligned;
        memcpy(output + (pixel - half_kernel_len) * pixel_stride, writeptr, n_times * sizeof(float));
    }

    // right border
    for (unsigned pixel = n_pixels - half_kernel_len; pixel < n_pixels; ++pixel) {
        const float *inptr = input + pixel * pixel_stride;
        const unsigned tmpidx = pixel % (half_kernel_len + 1);
        float *tmpptr = test + tmpidx * n_dims_aligned;

        unsigned dim;
        for (dim = 0; dim < dim_avx_end; dim += 8) {
            __m256 pixels = _mm256_loadu_ps(inptr + dim);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);
                __m256 pixel_mirrored;

                if (pixel + i < n_pixels)
                    pixel_mirrored = _mm256_loadu_ps(input + (pixel + i) * pixel_stride + dim);
                else
                    pixel_mirrored =
                        _mm256_loadu_ps(input + (n_pixels - ((i + pixel) % n_pixels) - 2) * pixel_stride + dim);

                if (is_symmetric)
                    pixels = _mm256_add_ps(pixel_mirrored, _mm256_loadu_ps(input + (pixel - i) * pixel_stride + dim));
                else
                    pixels = _mm256_sub_ps(pixel_mirrored, _mm256_loadu_ps(input + (pixel - i) * pixel_stride + dim));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        if (dim_left > 0) {
            __m256 pixels = _mm256_maskload_ps(inptr + dim, mask);
            __m256 kernel_val = _mm256_broadcast_ss(&kernel.coefs[0]);
            __m256 result = _mm256_mul_ps(pixels, kernel_val);

            for (unsigned int i = 1; i <= half_kernel_len; ++i) {
                kernel_val = _mm256_broadcast_ss(&kernel.coefs[i]);
                __m256 pixel_mirrored;

                if (pixel + i < n_pixels)
                    pixel_mirrored = _mm256_maskload_ps(input + (pixel + i) * pixel_stride + dim, mask);
                else
                    pixel_mirrored = _mm256_maskload_ps(
                        input + (n_pixels - ((i + pixel) % n_pixels) - 2) * pixel_stride + dim, mask);

                if (is_symmetric)
                    pixels = _mm256_add_ps(pixel_mirrored,
                                           _mm256_maskload_ps(input + (pixel - i) * pixel_stride + dim, mask));
                else
                    pixels = _mm256_sub_ps(pixel_mirrored,
                                           _mm256_maskload_ps(input + (pixel - i) * pixel_stride + dim, mask));
                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_store_ps(tmpptr + dim, result);
        }

        const unsigned writeidx = (pixel + 1) % (half_kernel_len + 1);
        float *writeptr = test + writeidx * n_dims_aligned;
        memcpy(output + (pixel - half_kernel_len) * pixel_stride, writeptr, n_times * sizeof(float));
    }

    for (unsigned i = 0; i < half_kernel_len; ++i) {
        unsigned pixel = n_pixels + i;
        const unsigned writeidx = (pixel + 1) % (half_kernel_len + 1);
        float *writeptr = test + writeidx * n_dims_aligned;
        memcpy(output + (pixel - half_kernel_len) * pixel_stride, writeptr, n_times * sizeof(float));
    }

    detail::avx_free(test);
}

void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned pixel_stride,
                                   const unsigned n_times, float *output, Kernel &kernel)
{
    if (kernel.is_symmetric)
        internal_convolve_fir_outer_single_avx<true>(input, n_pixels, pixel_stride, n_times, output, kernel);
    else
        internal_convolve_fir_outer_single_avx<false>(input, n_pixels, pixel_stride, n_times, output, kernel);
}

} // namespace detail

} // namespace fastfilters