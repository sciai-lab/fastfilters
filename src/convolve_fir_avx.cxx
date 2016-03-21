#include "fastfilters.hxx"

#include <immintrin.h>
#include <stdlib.h>

#include <stdexcept>

namespace fastfilters
{

namespace fir
{

template <bool is_symmetric>
static void internal_convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels,
                                                   const unsigned n_times, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();
    const unsigned int half_kernel_len = (kernel_len - 1) / 2;
    const unsigned int avx_end = (n_pixels - kernel_len) & ~31;
    const unsigned int avx_end_single = (n_pixels - kernel_len) & ~7;

    for (unsigned int dim = 0; dim < n_times; ++dim) {

        // take next line of pixels
        const float *cur_input = input + dim * n_pixels;
        float *cur_output = output + dim * n_pixels;

        // this function is only used for small kernels (<25 pixel)
        // such that non-vectorized code can be used for the border
        // treament without speed penalties
        unsigned int i = 0;
        for (i = 0; i < half_kernel_len; ++i) {
            float sum = 0.0;
            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                if ((int)i + kreal < 0)
                    sum += kernel[k] * cur_input[i - kreal];
                else
                    sum += kernel[k] * cur_input[i + kreal];
            }

            cur_output[i] = sum;
        }

        // working on 32 pixels at the same time leads to a fully used ymm register bank
        for (; i < avx_end; i += 32) {
            // load next 32 pixels
            __m256 result0 = _mm256_loadu_ps(cur_input + i);
            __m256 result1 = _mm256_loadu_ps(cur_input + i + 8);
            __m256 result2 = _mm256_loadu_ps(cur_input + i + 16);
            __m256 result3 = _mm256_loadu_ps(cur_input + i + 24);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_set1_ps(kernel[half_kernel_len]);
            result0 = _mm256_mul_ps(result0, kernel_val);
            result1 = _mm256_mul_ps(result1, kernel_val);
            result2 = _mm256_mul_ps(result2, kernel_val);
            result3 = _mm256_mul_ps(result3, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_set1_ps(kernel[half_kernel_len + j]);

                // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] = (image[i-j] +
                // image[i+j]) * kernel[j])
                // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                __m256 pixels0, pixels1, pixels2, pixels3;

                if (is_symmetric) {
                    pixels0 = _mm256_add_ps(_mm256_loadu_ps(cur_input + i + j), _mm256_loadu_ps(cur_input + i - j));
                    pixels1 =
                        _mm256_add_ps(_mm256_loadu_ps(cur_input + i + j + 8), _mm256_loadu_ps(cur_input + i - j + 8));
                    pixels2 =
                        _mm256_add_ps(_mm256_loadu_ps(cur_input + i + j + 16), _mm256_loadu_ps(cur_input + i - j + 16));
                    pixels3 =
                        _mm256_add_ps(_mm256_loadu_ps(cur_input + i + j + 24), _mm256_loadu_ps(cur_input + i - j + 24));
                } else {
                    pixels0 = _mm256_sub_ps(_mm256_loadu_ps(cur_input + i + j), _mm256_loadu_ps(cur_input + i - j));
                    pixels1 =
                        _mm256_sub_ps(_mm256_loadu_ps(cur_input + i + j + 8), _mm256_loadu_ps(cur_input + i - j + 8));
                    pixels2 =
                        _mm256_sub_ps(_mm256_loadu_ps(cur_input + i + j + 16), _mm256_loadu_ps(cur_input + i - j + 16));
                    pixels3 =
                        _mm256_sub_ps(_mm256_loadu_ps(cur_input + i + j + 24), _mm256_loadu_ps(cur_input + i - j + 24));
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
            __m256 result = _mm256_loadu_ps(cur_input + i);
            __m256 kernel_val = _mm256_set1_ps(kernel[half_kernel_len]);
            result = _mm256_mul_ps(result, kernel_val);

            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_set1_ps(kernel[half_kernel_len + j]);
                __m256 pixels;

                if (is_symmetric)
                    pixels = _mm256_add_ps(_mm256_loadu_ps(cur_input + i + j), _mm256_loadu_ps(cur_input + i - j));
                else
                    pixels = _mm256_sub_ps(_mm256_loadu_ps(cur_input + i + j), _mm256_loadu_ps(cur_input + i - j));

                result = _mm256_fmadd_ps(pixels, kernel_val, result);
            }

            _mm256_storeu_ps(cur_output + i, result);
        }

        // right border
        for (; i < n_pixels; ++i) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                if (kreal + i >= n_pixels)
                    sum += kernel[k] * cur_input[i - kreal];
                else
                    sum += kernel[k] * cur_input[i + kreal];
            }

            cur_output[i] = sum;
        }
    }
}

void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, Kernel &kernel)
{
    if (kernel.is_symmetric)
        internal_convolve_fir_inner_single_avx<true>(input, n_pixels, n_times, output, kernel);
    else
        internal_convolve_fir_inner_single_avx<false>(input, n_pixels, n_times, output, kernel);
}

template <bool is_symmetric>
static void internal_convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels,
                                                   const unsigned n_times, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();
    const unsigned int half_kernel_len = (kernel_len - 1) / 2;
    const unsigned int dim_avx_end = n_times & ~7;
    const unsigned int dim_left = n_times - dim_avx_end;
    const unsigned int pixels_avx_end = (n_pixels - kernel_len) & ~3;

    unsigned int dim;
    for (dim = 0; dim < dim_avx_end; dim += 8) {

        unsigned int i = 0;
        // left border - work on eight dimensions at once
        for (i = 0; i < half_kernel_len; ++i) {
            __m256 sum = _mm256_setzero_ps();

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                __m256 kernel_val = _mm256_set1_ps(kernel[k]);
                __m256 input_val;

                if ((int)i + kreal < 0)
                    input_val = _mm256_loadu_ps(input + dim + (i - kreal) * n_pixels);
                else
                    input_val = _mm256_loadu_ps(input + dim + (i + kreal) * n_pixels);

                sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
            }

            _mm256_storeu_ps(output + dim + i * n_pixels, sum);
        }

        // work on four pixels in eight dimensions at once to fill the ymm register bank
        for (; i < pixels_avx_end; i += 4) {
            __m256 result0 = _mm256_loadu_ps(input + dim + i * n_pixels);
            __m256 result1 = _mm256_loadu_ps(input + dim + (i + 1) * n_pixels);
            __m256 result2 = _mm256_loadu_ps(input + dim + (i + 2) * n_pixels);
            __m256 result3 = _mm256_loadu_ps(input + dim + (i + 3) * n_pixels);

            // multiply current pixels with center value of kernel
            __m256 kernel_val = _mm256_set1_ps(kernel[half_kernel_len]);
            result0 = _mm256_mul_ps(result0, kernel_val);
            result1 = _mm256_mul_ps(result1, kernel_val);
            result2 = _mm256_mul_ps(result2, kernel_val);
            result3 = _mm256_mul_ps(result3, kernel_val);

            // work on both sides of symmetric kernel simultaneously
            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_set1_ps(kernel[half_kernel_len + j]);

                // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] = (image[i-j] +
                // image[i+j]) * kernel[j])
                // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
                __m256 pixels0, pixels1, pixels2, pixels3;

                if (is_symmetric) {
                    pixels0 = _mm256_add_ps(_mm256_loadu_ps(input + dim + (i + j + 0) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 0) * n_pixels));
                    pixels1 = _mm256_add_ps(_mm256_loadu_ps(input + dim + (i + j + 1) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 1) * n_pixels));
                    pixels2 = _mm256_add_ps(_mm256_loadu_ps(input + dim + (i + j + 2) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 2) * n_pixels));
                    pixels3 = _mm256_add_ps(_mm256_loadu_ps(input + dim + (i + j + 3) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 3) * n_pixels));
                } else {
                    pixels0 = _mm256_sub_ps(_mm256_loadu_ps(input + dim + (i + j + 0) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i + j + 0) * n_pixels));
                    pixels1 = _mm256_sub_ps(_mm256_loadu_ps(input + dim + (i + j + 1) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 1) * n_pixels));
                    pixels2 = _mm256_sub_ps(_mm256_loadu_ps(input + dim + (i + j + 2) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 2) * n_pixels));
                    pixels3 = _mm256_sub_ps(_mm256_loadu_ps(input + dim + (i + j + 3) * n_pixels),
                                            _mm256_loadu_ps(input + dim + (i - j + 3) * n_pixels));
                }

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
                result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
                result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
                result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
            }

            // write result to output array
            _mm256_storeu_ps(output + dim + i * n_pixels, result0);
            _mm256_storeu_ps(output + dim + (i + 1) * n_pixels, result1);
            _mm256_storeu_ps(output + dim + (i + 2) * n_pixels, result2);
            _mm256_storeu_ps(output + dim + (i + 3) * n_pixels, result3);
        }

        // right border - eight dimensions at once
        for (; i < n_pixels; ++i) {
            __m256 sum = _mm256_setzero_ps();

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                __m256 input_val;
                __m256 kernel_val = _mm256_set1_ps(kernel[k]);

                if (kreal + i >= n_pixels)
                    input_val = _mm256_loadu_ps(input + dim + (i - kreal) * n_pixels);
                else
                    input_val = _mm256_loadu_ps(input + dim + (i + kreal) * n_pixels);

                sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
            }

            _mm256_storeu_ps(output + dim + i * n_pixels, sum);
        }
    }

    if (dim_left == 0)
        return;

    // work on up to seven missing dimensions
    // only difference to previous code is the use of masked load/store AVX2 instructions
    unsigned int i = 0;
    const __m256i mask = _mm256_set_epi32(0, dim_left >= 7 ? 0xffffffff : 0, dim_left >= 6 ? 0xffffffff : 0,
                                          dim_left >= 5 ? 0xffffffff : 0, dim_left >= 4 ? 0xffffffff : 0,
                                          dim_left >= 3 ? 0xffffffff : 0, dim_left >= 2 ? 0xffffffff : 0, 0xffffffff);

    // left border
    for (i = 0; i < half_kernel_len; ++i) {
        __m256 sum = _mm256_setzero_ps();

        for (unsigned int k = 0; k < kernel_len; ++k) {
            const int kreal = k - half_kernel_len;
            __m256 kernel_val = _mm256_set1_ps(kernel[k]);
            __m256 input_val;

            if ((int)i + kreal < 0)
                input_val = _mm256_maskload_ps(input + dim + (i - kreal) * n_pixels, mask);
            else
                input_val = _mm256_maskload_ps(input + dim + (i + kreal) * n_pixels, mask);

            sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
        }

        _mm256_maskstore_ps(output + dim + i * n_pixels, mask, sum);
    }

    // work on four pixels in up to eight dimensions at once to fill the ymm register bank
    for (; i < pixels_avx_end; i += 4) {
        __m256 result0 = _mm256_maskload_ps(input + dim + i * n_pixels, mask);
        __m256 result1 = _mm256_maskload_ps(input + dim + (i + 1) * n_pixels, mask);
        __m256 result2 = _mm256_maskload_ps(input + dim + (i + 2) * n_pixels, mask);
        __m256 result3 = _mm256_maskload_ps(input + dim + (i + 3) * n_pixels, mask);

        // multiply current pixels with center value of kernel
        __m256 kernel_val = _mm256_set1_ps(kernel[half_kernel_len]);
        result0 = _mm256_mul_ps(result0, kernel_val);
        result1 = _mm256_mul_ps(result1, kernel_val);
        result2 = _mm256_mul_ps(result2, kernel_val);
        result3 = _mm256_mul_ps(result3, kernel_val);

        // work on both sides of symmetric kernel simultaneously
        for (unsigned int j = 1; j <= half_kernel_len; ++j) {
            kernel_val = _mm256_set1_ps(kernel[half_kernel_len + j]);

            // sum pixels for both sides of kernel (kernel[-j] * image[i-j] + kernel[j] * image[i+j] = (image[i-j] +
            // image[i+j]) * kernel[j])
            // since kernel[-j] = kernel[j] or kernel[-j] = -kernel[j]
            __m256 pixels0, pixels1, pixels2, pixels3;

            if (is_symmetric) {
                pixels0 = _mm256_add_ps(_mm256_maskload_ps(input + dim + (i + j + 0) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 0) * n_pixels, mask));
                pixels1 = _mm256_add_ps(_mm256_maskload_ps(input + dim + (i + j + 1) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 1) * n_pixels, mask));
                pixels2 = _mm256_add_ps(_mm256_maskload_ps(input + dim + (i + j + 2) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 2) * n_pixels, mask));
                pixels3 = _mm256_add_ps(_mm256_maskload_ps(input + dim + (i + j + 3) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 3) * n_pixels, mask));
            } else {
                pixels0 = _mm256_sub_ps(_mm256_maskload_ps(input + dim + (i + j + 0) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i + j + 0) * n_pixels, mask));
                pixels1 = _mm256_sub_ps(_mm256_maskload_ps(input + dim + (i + j + 1) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 1) * n_pixels, mask));
                pixels2 = _mm256_sub_ps(_mm256_maskload_ps(input + dim + (i + j + 2) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 2) * n_pixels, mask));
                pixels3 = _mm256_sub_ps(_mm256_maskload_ps(input + dim + (i + j + 3) * n_pixels, mask),
                                        _mm256_maskload_ps(input + dim + (i - j + 3) * n_pixels, mask));
            }

            // multiply with kernel value and add to result
            result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
            result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
            result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
            result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
        }

        // write result to output array
        _mm256_maskstore_ps(output + dim + i * n_pixels, mask, result0);
        _mm256_maskstore_ps(output + dim + (i + 1) * n_pixels, mask, result1);
        _mm256_maskstore_ps(output + dim + (i + 2) * n_pixels, mask, result2);
        _mm256_maskstore_ps(output + dim + (i + 3) * n_pixels, mask, result3);
    }

    // right border
    for (; i < n_pixels; ++i) {
        __m256 sum = _mm256_setzero_ps();

        for (unsigned int k = 0; k < kernel_len; ++k) {
            const int kreal = k - half_kernel_len;
            __m256 input_val;
            __m256 kernel_val = _mm256_set1_ps(kernel[k]);

            if (kreal + i >= n_pixels)
                input_val = _mm256_maskload_ps(input + dim + (i - kreal) * n_pixels, mask);
            else
                input_val = _mm256_maskload_ps(input + dim + (i + kreal) * n_pixels, mask);

            sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
        }

        _mm256_maskstore_ps(output + dim + i * n_pixels, mask, sum);
    }
}

void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, Kernel &kernel)
{
    if (kernel.is_symmetric)
        internal_convolve_fir_outer_single_avx<true>(input, n_pixels, n_times, output, kernel);
    else
        internal_convolve_fir_outer_single_avx<false>(input, n_pixels, n_times, output, kernel);
}

} // namespace detail

} // namespace fastfilters