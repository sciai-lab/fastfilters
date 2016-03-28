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
