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
#ifndef FASTFILTERS_OPENCV_HXX
#define FASTFILTERS_OPENCV_HXX

#include <fastfilters.hxx>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace fastfilters
{

void sepFilter2D(cv::Mat &in, cv::Mat &out, fir::Kernel &kx, fir::Kernel &ky)
{
#if 0
    if (in.dims != 2)
        throw std::invalid_argument("Input array does not have two dimensions.");
    if (out.dims != 2)
        throw std::invalid_argument("Output array does not have two dimensions.");
    if (!in.isContinuous())
        throw std::invalid_argument("Input array is not continuous.");
    if (!out.isContinuous())
        throw std::invalid_argument("Output array is not continuous.");
    if (in.rows != out.rows || in.cols != out.cols)
        throw std::invalid_argument("Input and output dimensions don't match.");
#endif

    const float *inptr = (float *)in.data;
    float *outptr = (float *)out.data;

    fir::convolve_fir(inptr, in.rows, in.cols, in.cols, 1, outptr, ky);
    fir::convolve_fir(outptr, in.cols, 1, in.rows, in.cols, outptr, kx);
}

} // namespace fastfilters
#endif
