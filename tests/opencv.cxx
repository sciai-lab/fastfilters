#include <opencv2/opencv.hpp>
#include <fastfilters/opencv.hxx>

#include <iostream>
#include <ctime>

void test()
{
    const unsigned int size = 10000;

    std::vector<float> kel = {3, 2, 1, 0, 1, 2, 3};
    std::vector<float> kel2 = {0, 1, 2, 3};
    cv::Mat M(size, size, CV_32F, cv::Scalar::all(0.0));
    cv::Mat M2(size, size, CV_32F, cv::Scalar::all(0.0));
    cv::Mat Nopencv(size, size, CV_32F, cv::Scalar::all(0.0));
    cv::Mat Nff(size, size, CV_32F, cv::Scalar::all(0.0));
    cv::Mat kernel(7, 1, CV_32F, cv::Scalar::all(0.0));
    fastfilters::fir::Kernel kern(true, kel2);

    for (unsigned int i = 0; i < 7; ++i)
        kernel.at<float>(i, 0) = kel[i];

    std::clock_t opencv_start = std::clock();
    cv::sepFilter2D(M, Nopencv, CV_32F, kernel, kernel.t(), cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    std::clock_t opencv_end = std::clock();
    double opencv_time = opencv_end - opencv_start;

    std::cout << "  OpenCV: " << opencv_time << "\n";

    std::clock_t ff_start = std::clock();
    fastfilters::sepFilter2D(M2, Nff, kern, kern);
    std::clock_t ff_end = std::clock();
    double ff_time = ff_end - ff_start;

    std::cout << "  FastFilters: " << ff_time << "\n";
    std::cout << "faster by: " << opencv_time / ff_time << "\n";
}

int main()
{
    test();
    return -1;
}