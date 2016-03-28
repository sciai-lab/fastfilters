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
#include "convolve_fir.hxx"

namespace fastfilters
{
namespace fir
{

void convolve_fir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times,
                               const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (detail::cpu_has_avx_fma())
        convolve_fir_inner_single_avx_fma(input, n_pixels, n_times, dim_stride, output, kernel);
    else if(detail::cpu_has_avx())
        convolve_fir_inner_single_avx(input, n_pixels, n_times, dim_stride, output, kernel);
    else
        convolve_fir_inner_single_noavx(input, n_pixels, n_times, dim_stride, output, kernel);
}

void convolve_fir_outer_single(const float *input, const unsigned int n_pixels, const unsigned int pixel_stride,
                               const unsigned n_times, const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (detail::cpu_has_avx_fma() && dim_stride == 1)
        convolve_fir_outer_single_avx_fma(input, n_pixels, pixel_stride, n_times, output, kernel);
    else if (detail::cpu_has_avx() && dim_stride == 1)
        convolve_fir_outer_single_avx(input, n_pixels, pixel_stride, n_times, output, kernel);
    else
        convolve_fir_outer_single_noavx(input, n_pixels, pixel_stride, n_times, dim_stride, output, kernel);
}

void convolve_fir(const float *input, const unsigned int pixel_n, const unsigned int pixel_stride,
                  const unsigned int dim_n, const unsigned int dim_stride, float *output, Kernel &kernel)
{
    if (pixel_stride == 1)
        convolve_fir_inner_single(input, pixel_n, dim_n, dim_stride, output, kernel);
    else
        convolve_fir_outer_single(input, pixel_n, pixel_stride, dim_n, dim_stride, output, kernel);
}

} // namespace fir

} // namespace fastfilters
