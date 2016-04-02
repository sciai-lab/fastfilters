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
#include "fastfilters.hxx"
#include "vector.hxx"
#include <iostream>

namespace fastfilters
{

namespace fir
{

namespace
{

template <unsigned half_kernel_len>
static void internal_convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels,
                                                     const unsigned n_times, const unsigned int dim_stride,
                                                     float *output, Kernel &kernel)
{
    const unsigned int kernel_len = 2 * half_kernel_len + 1;
    ConstantVector<float> tmpline(n_pixels);

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * dim_stride;
        float *cur_output = output + i * dim_stride;

        for (unsigned int j = 0; j < n_pixels; ++j)
            tmpline[j] = cur_input[j];

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + (int)j < 0)
                    offset = -j - kreal;
                else
                    offset = j + kreal;
                sum += kernel[k] * tmpline[offset];
            }

            cur_output[j] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * tmpline[j + kreal];
            }

            cur_output[j] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;
            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + j >= n_pixels)
                    offset = n_pixels - ((kreal + j) % n_pixels) - 2;
                else
                    offset = j + kreal;
                sum += kernel[k] * tmpline[offset];
            }

            cur_output[j] = sum;
        }
    }
}

template <unsigned half_kernel_len>
static void internal_convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels,
                                                     const unsigned int pixel_stride, const unsigned n_times,
                                                     const unsigned dim_stride, float *output, Kernel &kernel)
{
    // const unsigned int kernel_len = kernel.len();
    const unsigned int kernel_len = 2 * half_kernel_len + 1;
    ConstantVector<float> tmpline(n_pixels);

    for (unsigned int i = 0; i < n_times; ++i) {

        // take next line of pixels
        const float *cur_input = input + i * dim_stride;
        float *cur_output = output + i * dim_stride;

        for (unsigned int j = 0; j < n_pixels; ++j)
            tmpline[j] = cur_input[j * pixel_stride];

        // left border
        unsigned int j = 0;
        for (j = 0; j < kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + (int)j < 0)
                    offset = -j - kreal;
                else
                    offset = j + kreal;
                sum += kernel[k] * tmpline[offset];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // full line
        for (; j < n_pixels - kernel_len / 2; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                sum += kernel[k] * tmpline[(j + kreal)];
            }

            cur_output[j * pixel_stride] = sum;
        }

        // right border
        for (; j < n_pixels; ++j) {
            float sum = 0.0;

            for (unsigned int k = 0; k < kernel_len; ++k) {
                const int kreal = k - kernel_len / 2;
                unsigned int offset;
                if (kreal + j >= n_pixels)
                    offset = n_pixels - ((kreal + j) % n_pixels) - 2;
                else
                    offset = j + kreal;
                sum += kernel[k] * tmpline[offset];
            }

            cur_output[j * pixel_stride] = sum;
        }
    }
}

} // anonymous namespace

void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                     const unsigned int dim_stride, float *output, Kernel &kernel)
{
    switch (kernel.half_len()) {
    case 1:
        internal_convolve_fir_inner_single_noavx<1>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 2:
        internal_convolve_fir_inner_single_noavx<2>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 3:
        internal_convolve_fir_inner_single_noavx<3>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 4:
        internal_convolve_fir_inner_single_noavx<4>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 5:
        internal_convolve_fir_inner_single_noavx<5>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 6:
        internal_convolve_fir_inner_single_noavx<6>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 7:
        internal_convolve_fir_inner_single_noavx<7>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 8:
        internal_convolve_fir_inner_single_noavx<8>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 9:
        internal_convolve_fir_inner_single_noavx<9>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 10:
        internal_convolve_fir_inner_single_noavx<10>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 11:
        internal_convolve_fir_inner_single_noavx<11>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    case 12:
        internal_convolve_fir_inner_single_noavx<12>(input, n_pixels, n_times, dim_stride, output, kernel);
        break;
    default:
        throw std::logic_error("Kernel too long.");
    }
}

void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                                     const unsigned n_times, const unsigned dim_stride, float *output, Kernel &kernel)
{
    switch (kernel.half_len()) {
    case 1:
        internal_convolve_fir_outer_single_noavx<1>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 2:
        internal_convolve_fir_outer_single_noavx<2>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 3:
        internal_convolve_fir_outer_single_noavx<3>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 4:
        internal_convolve_fir_outer_single_noavx<4>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 5:
        internal_convolve_fir_outer_single_noavx<5>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 6:
        internal_convolve_fir_outer_single_noavx<6>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 7:
        internal_convolve_fir_outer_single_noavx<7>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 8:
        internal_convolve_fir_outer_single_noavx<8>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 9:
        internal_convolve_fir_outer_single_noavx<9>(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
        break;
    case 10:
        internal_convolve_fir_outer_single_noavx<10>(input, n_pixels, pixel_stride, n_times, dim_stride, output,
                                                     kernel);
        break;
    case 11:
        internal_convolve_fir_outer_single_noavx<11>(input, n_pixels, pixel_stride, n_times, dim_stride, output,
                                                     kernel);
        break;
    case 12:
        internal_convolve_fir_outer_single_noavx<12>(input, n_pixels, pixel_stride, n_times, dim_stride, output,
                                                     kernel);
        break;
    default:
        throw std::logic_error("Kernel too long.");
    }
}

} // namespace detail

} // namespace fastfilters
