#include "fastfilters.hxx"

#include <iostream>
#include <fstream>
#include <string.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <iomanip>

int main()
{
    std::vector<float> input(10 * 512);
    std::vector<float> output(10 * 512);
    std::vector<float> result(10 * 512);
    std::cout << std::setprecision(30);

    for (unsigned int i = 0; i < 10; ++i)
        input[512 * i + 256] = 1.0;

    fastfilters::iir::Coefficients coefs(5.0, 0);
    fastfilters::iir::convolve_iir_inner_single_avx(input.data(), 512, 10, output.data(), coefs);

    std::ifstream ifp1("tests/iir_single_avx0.b", std::ios::in | std::ios::binary);
    ifp1.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    ifp1.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(result[0]));
    ifp1.close();

    for (unsigned i = 0; i < 512; ++i) {
        for (unsigned int j = 0; j < 9; ++j) {
            float diff = std::abs(output[512 * j + i] - result[i]);
            assert(diff < 5 * 1e-6);
        }
    }

    std::vector<float> input2(512 * 50);
    std::vector<float> output2(512 * 50);
    std::vector<float> result2(512 * 50);

    for (unsigned int i = 0; i < 50; ++i)
        input2[256 * 50 + i] = 1.0;

    fastfilters::iir::convolve_iir_outer_single_avx(input2.data(), 512, 50, output2.data(), coefs);

    std::ifstream ifp("tests/iir_single_avx1.b", std::ios::in | std::ios::binary);
    ifp.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    ifp.read(reinterpret_cast<char *>(result2.data()), result2.size() * sizeof(result2[0]));
    ifp.close();

    for (unsigned i = 0; i < 512; ++i) {
        for (unsigned j = 0; j < 50; ++j) {
            float diff = std::abs(output2[i * 50 + j] - result2[i * 50 + j]);
            assert(diff < 5 * 1e-6);
        }
    }

    return 0;
}