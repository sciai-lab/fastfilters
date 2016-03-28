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
#ifndef FASTFILTERS_VIGRA_HXX
#define FASTFILTERS_VIGRA_HXX

#include "fastfilters.hxx"
#include <vigra/multi_array.hxx>

namespace fastfilters
{

template <unsigned ndim>
inline void separableConvolveMultiArray(const vigra::MultiArrayView<ndim, float> input,
                                        vigra::MultiArrayView<ndim, float> &output, fir::Kernel &kernel)
{
    const float *inptr = input.data();
    float *outptr = output.data();

    if (ndim == 1) {
        fastfilters::fir::convolve_fir(inptr, input.size(), 1, 1, 1, outptr, kernel);
        return;
    }

    // outermost dimension
    unsigned int n_times = 1;
    for (unsigned int j = 1; j < ndim; ++j) {
        n_times *= input.shape()[j];
    }
    fastfilters::fir::convolve_fir(inptr, input.shape()[0], n_times, n_times, 1, outptr, kernel);

    if (ndim > 2) {
        for (unsigned int i = ndim - 2; i > 0; --i) {
            unsigned int n_times_inner = 1;
            unsigned int n_time_outer = 1;
            for (unsigned int j = 0; j < i; ++j)
                n_time_outer *= input.shape()[j];
            for (unsigned int j = i + 1; j < ndim; ++j)
                n_times_inner *= input.shape()[j];

            for (unsigned int j = 0; j < n_time_outer; ++j) {
                fastfilters::fir::convolve_fir(outptr + n_times_inner * input.shape()[i - 1] * j, input.shape()[i],
                                               n_times_inner, n_times_inner, 1,
                                               outptr + n_times_inner * input.shape()[i - 1] * j, kernel);
            }
        }
    }

    // innermost dimension
    n_times = 1;
    for (unsigned int i = 0; i < ndim - 1; ++i)
        n_times *= input.shape()[i];
    fastfilters::fir::convolve_fir(outptr, input.shape()[ndim - 1], 1, n_times, input.shape()[ndim - 1], outptr,
                                   kernel);
}

} // namespace fastfilters
#endif
