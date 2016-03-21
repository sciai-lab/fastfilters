#include "fastfilters.hxx"

namespace fastfilters
{

namespace detail
{

void convolve_fir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const float *kernel, const unsigned int kernel_len)
{
    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * n_pixels;
        float *cur_output = output + i * n_pixels;

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

void convolve_fir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const float *kernel, const unsigned int kernel_len)
{
    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i;
        float *cur_output = output + i;

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + (int)j < 0)
                    sum += kernel[k] * cur_input[(j - kreal) * n_times];
                else
                    sum += kernel[k] * cur_input[(j + kreal) * n_times];
            }

            cur_output[j * n_times] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * cur_input[(j + kreal) * n_times];
            }

            cur_output[j * n_times] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                if (kreal + j >= n_pixels)
                    sum += kernel[k] * cur_input[(j - kreal) * n_times];
                else
                    sum += kernel[k] * cur_input[(j + kreal) * n_times];
            }

            cur_output[j * n_times] = sum;
        }
    }
}

} // namespace detail

} // namespace fastfilters