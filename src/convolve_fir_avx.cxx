#include "fastfilters.hxx"
#include "vector.hxx"

#include <immintrin.h>
#include <stdlib.h>

#include <stdexcept>

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
    ConstantVector<float> cur_input(n_pixels);

    float *tmp = NULL;
    int res = posix_memalign((void **)&tmp, 32, sizeof(float) * n_pixels);

    if (res < 0 || tmp == NULL)
        throw std::runtime_error("posix_memalign failed.");

    for (unsigned int dim = 0; dim < n_times; ++dim) {

        // take next line of pixels
        // const float *cur_input = input + ;
        float *cur_output = output + dim * dim_stride;

        for (unsigned int j = 0; j < (n_pixels & ~7); j += 8)
            _mm256_store_ps(tmp + j, _mm256_loadu_ps(input + dim * dim_stride + j));
        for (unsigned int j = (n_pixels & ~7); j < n_pixels; ++j)
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
            __m256 kernel_val = _mm256_set1_ps(kernel[half_kernel_len]);
            result = _mm256_mul_ps(result, kernel_val);

            for (unsigned int j = 1; j <= half_kernel_len; ++j) {
                kernel_val = _mm256_set1_ps(kernel[half_kernel_len + j]);
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

    free(tmp);
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
    const unsigned int kernel_len = kernel.len();
    const unsigned int half_kernel_len = kernel.half_len();
    const unsigned int dim_avx_end = n_times & ~7;
    const unsigned int dim_left = n_times - dim_avx_end;
    const unsigned int pixels_avx_end = (n_pixels - kernel_len) & ~3;

    const unsigned int n_pixels_aligned = n_pixels;
    float *tmp = NULL;
    int res = posix_memalign((void **)&tmp, 32, sizeof(float) * n_pixels_aligned * 8 * 4);

    if (res < 0 || tmp == NULL)
        throw std::runtime_error("posix_memalign failed.");

    unsigned int dim;
    for (dim = 0; dim < dim_avx_end; dim += 8) {

        unsigned int i = 0;
        for (i = 0; i < n_pixels; ++i)
            _mm256_store_ps(tmp + 8 * i, _mm256_loadu_ps(input + dim + i * pixel_stride));

        i = 0;
        // left border - work on eight dimensions at once
        for (i = 0; i < half_kernel_len; ++i) {
            __m256 sum = _mm256_setzero_ps();

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                unsigned int offset;
                __m256 kernel_val = _mm256_set1_ps(kernel[k]);
                __m256 input_val;

                if (kreal + (int)i < 0)
                    offset = -i - kreal;
                else
                    offset = i + kreal;

                input_val = _mm256_load_ps(tmp + offset * 8);
                sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
            }

            _mm256_storeu_ps(output + dim + i * pixel_stride, sum);
        }

        // work on four pixels in eight dimensions at once to fill the ymm register bank
        for (; i < pixels_avx_end; i += 4) {
            __m256 result0 = _mm256_load_ps(tmp + i * 8);
            __m256 result1 = _mm256_load_ps(tmp + (i + 1) * 8);
            __m256 result2 = _mm256_load_ps(tmp + (i + 2) * 8);
            __m256 result3 = _mm256_load_ps(tmp + (i + 3) * 8);

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
                    pixels0 =
                        _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 0) * 8), _mm256_load_ps(tmp + (i - j + 0) * 8));
                    pixels1 =
                        _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 1) * 8), _mm256_load_ps(tmp + (i - j + 1) * 8));
                    pixels2 =
                        _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 2) * 8), _mm256_load_ps(tmp + (i - j + 2) * 8));
                    pixels3 =
                        _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 3) * 8), _mm256_load_ps(tmp + (i - j + 3) * 8));
                } else {
                    pixels0 =
                        _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 0) * 8), _mm256_load_ps(tmp + (i + j + 0) * 8));
                    pixels1 =
                        _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 1) * 8), _mm256_load_ps(tmp + (i - j + 1) * 8));
                    pixels2 =
                        _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 2) * 8), _mm256_load_ps(tmp + (i - j + 2) * 8));
                    pixels3 =
                        _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 3) * 8), _mm256_load_ps(tmp + (i - j + 3) * 8));
                }

                // multiply with kernel value and add to result
                result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
                result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
                result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
                result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
            }

            // write result to output array
            _mm256_storeu_ps(output + dim + i * pixel_stride, result0);
            _mm256_storeu_ps(output + dim + (i + 1) * pixel_stride, result1);
            _mm256_storeu_ps(output + dim + (i + 2) * pixel_stride, result2);
            _mm256_storeu_ps(output + dim + (i + 3) * pixel_stride, result3);
        }

        // right border - eight dimensions at once
        for (; i < n_pixels; ++i) {
            __m256 sum = _mm256_setzero_ps();

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - half_kernel_len;
                __m256 input_val;
                __m256 kernel_val = _mm256_set1_ps(kernel[k]);

                unsigned int offset;
                if (kreal + i >= n_pixels)
                    offset = n_pixels - ((kreal + i) % n_pixels) - 2;
                else
                    offset = i + kreal;

                input_val = _mm256_load_ps(tmp + offset * 8);

                sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
            }

            _mm256_storeu_ps(output + dim + i * pixel_stride, sum);
        }
    }

    if (dim_left == 0) {
        free(tmp);
        return;
    }

    // work on up to seven missing dimensions
    // only difference to previous code is the use of masked load/store AVX2 instructions
    unsigned int i = 0;
    const __m256i mask = _mm256_set_epi32(0, dim_left >= 7 ? 0xffffffff : 0, dim_left >= 6 ? 0xffffffff : 0,
                                          dim_left >= 5 ? 0xffffffff : 0, dim_left >= 4 ? 0xffffffff : 0,
                                          dim_left >= 3 ? 0xffffffff : 0, dim_left >= 2 ? 0xffffffff : 0, 0xffffffff);

    for (unsigned int i = 0; i < n_pixels; ++i)
        _mm256_store_ps(tmp + 8 * i, _mm256_maskload_ps(input + dim + i * pixel_stride, mask));

    // left border
    for (i = 0; i < half_kernel_len; ++i) {
        __m256 sum = _mm256_setzero_ps();

        for (unsigned int k = 0; k < kernel_len; ++k) {
            const int kreal = k - half_kernel_len;
            __m256 kernel_val = _mm256_set1_ps(kernel[k]);
            __m256 input_val;

            if ((int)i + kreal < 0)
                input_val = _mm256_load_ps(tmp + (-i - kreal) * 8);
            else
                input_val = _mm256_load_ps(tmp + (i + kreal) * 8);

            sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
        }

        _mm256_maskstore_ps(output + dim + i * pixel_stride, mask, sum);
    }

    // work on four pixels in up to eight dimensions at once to fill the ymm register bank
    for (; i < pixels_avx_end; i += 4) {
        __m256 result0 = _mm256_load_ps(tmp + i * 8);
        __m256 result1 = _mm256_load_ps(tmp + (i + 1) * 8);
        __m256 result2 = _mm256_load_ps(tmp + (i + 2) * 8);
        __m256 result3 = _mm256_load_ps(tmp + (i + 3) * 8);

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
                pixels0 = _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 0) * 8), _mm256_load_ps(tmp + (i - j + 0) * 8));
                pixels1 = _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 1) * 8), _mm256_load_ps(tmp + (i - j + 1) * 8));
                pixels2 = _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 2) * 8), _mm256_load_ps(tmp + (i - j + 2) * 8));
                pixels3 = _mm256_add_ps(_mm256_load_ps(tmp + (i + j + 3) * 8), _mm256_load_ps(tmp + (i - j + 3) * 8));
            } else {
                pixels0 = _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 0) * 8), _mm256_load_ps(tmp + (i + j + 0) * 8));
                pixels1 = _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 1) * 8), _mm256_load_ps(tmp + (i - j + 1) * 8));
                pixels2 = _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 2) * 8), _mm256_load_ps(tmp + (i - j + 2) * 8));
                pixels3 = _mm256_sub_ps(_mm256_load_ps(tmp + (i + j + 3) * 8), _mm256_load_ps(tmp + (i - j + 3) * 8));
            }

            // multiply with kernel value and add to result
            result0 = _mm256_fmadd_ps(pixels0, kernel_val, result0);
            result1 = _mm256_fmadd_ps(pixels1, kernel_val, result1);
            result2 = _mm256_fmadd_ps(pixels2, kernel_val, result2);
            result3 = _mm256_fmadd_ps(pixels3, kernel_val, result3);
        }

        // write result to output array
        _mm256_maskstore_ps(output + dim + i * pixel_stride, mask, result0);
        _mm256_maskstore_ps(output + dim + (i + 1) * pixel_stride, mask, result1);
        _mm256_maskstore_ps(output + dim + (i + 2) * pixel_stride, mask, result2);
        _mm256_maskstore_ps(output + dim + (i + 3) * pixel_stride, mask, result3);
    }

    // right border
    for (; i < n_pixels; ++i) {
        __m256 sum = _mm256_setzero_ps();

        for (unsigned int k = 0; k < kernel_len; ++k) {
            const int kreal = k - half_kernel_len;
            __m256 input_val;
            __m256 kernel_val = _mm256_set1_ps(kernel[k]);

            unsigned int offset;
            if (kreal + i >= n_pixels)
                offset = n_pixels - ((kreal + i) % n_pixels) - 2;
            else
                offset = i + kreal;

            input_val = _mm256_load_ps(tmp + offset * 8);
            sum = _mm256_fmadd_ps(input_val, kernel_val, sum);
        }

        _mm256_maskstore_ps(output + dim + i * pixel_stride, mask, sum);
    }

    free(tmp);
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