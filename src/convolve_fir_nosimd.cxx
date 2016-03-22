#include "fastfilters.hxx"

namespace fastfilters
{

namespace fir
{

void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     const unsigned int dim_stride, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * dim_stride;
        float *cur_output = output + i * dim_stride;

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + (int)j < 0)
                    sum += kernel[k] * cur_input[j - kreal];
                else
                    sum += kernel[k] * cur_input[j + kreal];
            }

            cur_output[j] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * cur_input[j + kreal];
            }

            cur_output[j] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + j >= n_pixels)
                    sum += kernel[k] * cur_input[j - kreal];
                else
                    sum += kernel[k] * cur_input[j + kreal];
            }

            cur_output[j] = sum;
        }
    }
}

void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                                     const unsigned n_times, const unsigned dim_stride, float *output, Kernel &kernel)
{
    const unsigned int kernel_len = kernel.len();

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i;
        float *cur_output = output + i * dim_stride;

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + (int)j < 0)
                    sum += kernel[k] * cur_input[(j - kreal) * pixel_stride];
                else
                    sum += kernel[k] * cur_input[(j + kreal) * pixel_stride];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * cur_input[(j + kreal) * pixel_stride];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + j >= n_pixels)
                    sum += kernel[k] * cur_input[(j - kreal) * pixel_stride];
                else
                    sum += kernel[k] * cur_input[(j + kreal) * pixel_stride];
            }

            cur_output[j * pixel_stride] = sum;
        }
    }
}

} // namespace detail

} // namespace fastfilters