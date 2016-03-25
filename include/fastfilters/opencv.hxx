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
