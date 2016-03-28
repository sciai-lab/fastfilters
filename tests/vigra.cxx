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
#include <vigra/multi_array.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>
#include <vigra/unittest.hxx>
#include <fastfilters.hxx>
#include <fastfilters/vigra.hxx>

#include <iostream>
#include <ctime>

template <unsigned ndim> void run_test()
{
    vigra::MultiArray<ndim, float> vigra_array;
    vigra::MultiArray<ndim, float> vigra_output;
    vigra::MultiArray<ndim, float> ff_output;
    typename vigra::MultiArray<ndim, float>::difference_type shape;

    std::vector<float> kernel_values = {0, 1, 2, 3};
    vigra::Kernel1D<float> vigra_kernel;
    fastfilters::fir::Kernel ff_kernel(true, kernel_values);

    for (unsigned int i = 0; i < ndim; ++i)
        shape[i] = ceil(pow(1000000, 1 / ((float)ndim)));

    std::cout << "Running speed test in " << ndim << " dimensions with len " << shape[0] << "\n";

    vigra_array.reshape(shape);
    vigra_output.reshape(shape);
    ff_output.reshape(shape);

    int size = vigra_array.size();
    for (int i = 0; i < size; ++i)
        vigra_array[i] = (float)vigra::randomMT19937().uniform();

    vigra_kernel.initExplicitly(-3, 3);
    vigra_kernel.setBorderTreatment(vigra::BORDER_TREATMENT_REFLECT);

    for (int i = 0; i < 7; ++i)
        vigra_kernel[i - 3] = ff_kernel[i];

    std::clock_t vigra_start = std::clock();
    separableConvolveMultiArray(vigra_array, vigra_output, vigra_kernel);
    std::clock_t vigra_end = std::clock();
    double vigra_time = vigra_end - vigra_start;

    std::cout << "  Vigra: " << vigra_time << "\n";

    std::clock_t ff_start = std::clock();
    fastfilters::separableConvolveMultiArray(vigra_array, ff_output, ff_kernel);
    std::clock_t ff_end = std::clock();
    double ff_time = ff_end - ff_start;

    std::cout << "  fastfilters: " << ff_time << "\n";
    std::cout << " faster by: " << vigra_time / ff_time << "\n";

    shouldEqualSequenceTolerance(vigra_output.begin(), vigra_output.end(), ff_output.begin(), 1e-6);
}

int main()
{
    run_test<1>();
    run_test<2>();
    run_test<3>();
    run_test<4>();
    run_test<5>();
    return 0;
}
