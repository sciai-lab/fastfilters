#include "fastfilters.hxx"

#include <iostream>
#include <fstream>
#include <string.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <iomanip>

void test_inner(void)
{
    std::vector<float> input(10 * 512);
    std::vector<float> output(10 * 512);
    std::vector<float> output_avx(10 * 512);
    std::vector<float> result(10 * 512);
    fastfilters::iir::Coefficients coefs(5.0, 0);

    std::ifstream ifp1("tests/iir_inner.b", std::ios::in | std::ios::binary);
    ifp1.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    ifp1.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(result[0]));
    ifp1.close();

    fastfilters::iir::convolve_iir_inner_single_noavx(input.data(), 512, 10, output.data(), coefs);

    if (fastfilters::detail::cpu_has_avx2())
        fastfilters::iir::convolve_iir_inner_single_avx(input.data(), 512, 10, output_avx.data(), coefs);

    for (unsigned i = 0; i < 512; ++i) {
        for (unsigned int j = 0; j < 9; ++j) {
            float diff = std::abs(output[512 * j + i] - result[512 * j + i]);
            assert(diff < 5 * 1e-6 && "test_inner");
        }
    }

    if (fastfilters::detail::cpu_has_avx2()) {
        for (unsigned i = 0; i < 512; ++i) {
            for (unsigned int j = 0; j < 9; ++j) {
                float diff = std::abs(output_avx[512 * j + i] - result[512 * j + i]);
                assert(diff < 5 * 1e-6 && "test_inner_avx");
            }
        }
    }
}

void test_outer(void)
{
    std::vector<float> input(512 * 50);
    std::vector<float> output(512 * 50);
    std::vector<float> output_avx(512 * 50);
    std::vector<float> result(512 * 50);
    fastfilters::iir::Coefficients coefs(5.0, 0);

    std::ifstream ifp("tests/iir_outer.b", std::ios::in | std::ios::binary);
    ifp.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    ifp.read(reinterpret_cast<char *>(result.data()), result.size() * sizeof(result[0]));
    ifp.close();

    for (unsigned int i = 0; i < 50; ++i)
        input[256 * 50 + i] = 1.0;

    fastfilters::iir::convolve_iir_outer_single_noavx(input.data(), 512, 50, output.data(), coefs, 50);

    if (fastfilters::detail::cpu_has_avx2())
        fastfilters::iir::convolve_iir_outer_single_avx(input.data(), 512, 50, output_avx.data(), coefs);

    for (unsigned i = 0; i < 512; ++i) {
        for (unsigned j = 0; j < 50; ++j) {
            float diff = std::abs(output[i * 50 + j] - result[i * 50 + j]);
            assert(diff < 5 * 1e-6 && "test_outer");
        }
    }

    if (fastfilters::detail::cpu_has_avx2()) {
        for (unsigned i = 0; i < 512; ++i) {
            for (unsigned j = 0; j < 50; ++j) {
                float diff = std::abs(output_avx[i * 50 + j] - result[i * 50 + j]);
                assert(diff < 5 * 1e-6 && "test_outer_avx");
            }
        }
    }
}

int main()
{
    std::cout << std::setprecision(30);

    test_inner();
    test_outer();

    return 0;
}