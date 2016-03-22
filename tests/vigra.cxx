#include <vigra/multi_array.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/random.hxx>
#include <fastfilters.hxx>
#include <fastfilters/vigra.hxx>

#include <iostream>
#include <ctime>

template <unsigned ndim> void run_test()
{
    vigra::MultiArray<ndim, float> vigra_array;
    typename vigra::MultiArray<ndim, float>::difference_type shape;

    std::vector<float> kernel_values = {0, 1, 2, 3};
    vigra::Kernel1D<float> vigra_kernel;
    fastfilters::fir::Kernel ff_kernel(true, kernel_values);

    std::cout << "Running speed test in " << ndim << " dimensions\n";

    for (unsigned int i = 0; i < ndim; ++i)
        shape[i] = 512;

    vigra_array.reshape(shape);

    int size = vigra_array.size();
    for (int i = 0; i < size; ++i)
        vigra_array[i] = (float)vigra::randomMT19937().uniform();

    vigra_kernel.initExplicitly(-3, 3);
    for (int i = 0; i <= 3; ++i)
        vigra_kernel[i] = vigra_kernel[-i] = kernel_values[i];

    std::clock_t vigra_start = std::clock();
    separableConvolveMultiArray(vigra_array, vigra_array, vigra_kernel);
    std::clock_t vigra_end = std::clock();
    double vigra_time = vigra_end - vigra_start;

    std::cout << "Vigra: " << vigra_time << "\n";

    std::clock_t ff_start = std::clock();
    fastfilters::separableConvolveMultiArray(vigra_array, vigra_array, ff_kernel);
    std::clock_t ff_end = std::clock();
    double ff_time = ff_end - ff_start;

    std::cout << "fastfilters: " << ff_time << "\n";
}

int main()
{
    run_test<1>();
    run_test<2>();
    run_test<3>();

    return -1;
}