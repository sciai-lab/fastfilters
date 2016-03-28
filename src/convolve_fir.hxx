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

#ifndef CONVOLVE_FIR_HXX
#define CONVOLVE_FIR_HXX

#include "fastfilters.hxx"

namespace fastfilters
{
namespace fir
{

FASTFILTERS_API_EXPORT void convolve_fir_inner_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned n_times, const unsigned int dim_stride,
                                                          float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single_avx(const float *input, const unsigned int n_pixels,
                                                          const unsigned int pixel_stride, const unsigned n_times,
                                                          float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_inner_single_avx_fma(const float *input, const unsigned int n_pixels,
                                                              const unsigned n_times, const unsigned int dim_stride,
                                                              float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single_avx_fma(const float *input, const unsigned int n_pixels,
                                                              const unsigned int pixel_stride, const unsigned n_times,
                                                              float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_inner_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned n_times, const unsigned int dim_stride,
                                                            float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single_noavx(const float *input, const unsigned int n_pixels,
                                                            const unsigned int pixel_stride, const unsigned n_times,
                                                            const unsigned dim_stride, float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_inner_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned n_times, const unsigned int dim_stride,
                                                      float *output, Kernel &kernel);

FASTFILTERS_API_EXPORT void convolve_fir_outer_single(const float *input, const unsigned int n_pixels,
                                                      const unsigned int pixel_stride, const unsigned n_times,
                                                      const unsigned int dim_stride, float *output, Kernel &kernel);
}
}
#endif