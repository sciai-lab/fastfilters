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
#include "vector.hxx"
#include "util.hxx"

#include <immintrin.h>
#include <stdlib.h>

#include <stdexcept>

#define CONVOLVE_IIR_FUNCTION(x) static void optimized_##x
#include "convolve_iir.cxx"

namespace fastfilters
{

namespace iir
{

void convolve_iir_inner_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs)
{
    __m256 mm_n_causal[4];
    __m256 mm_n_anticausal[4];
    __m256 mm_d[4];

    const unsigned n_times_avx = n_times & ~7;
    const unsigned n_times_normal = n_times - n_times_avx;

    for (unsigned int i = 0; i < 4; ++i) {
        mm_n_causal[i] = _mm256_set1_ps(coefs.n_causal[i]);
        mm_n_anticausal[i] = _mm256_set1_ps(coefs.n_anticausal[i]);
        mm_d[i] = _mm256_set1_ps(-coefs.d[i]);
    }

    float *tmp = (float *)detail::avx_memalign(sizeof(float) * n_pixels * 8);

    for (unsigned int dim = 0; dim < n_times_avx; dim += 8) {
        __m256 prev_in0, prev_in1, prev_in2, prev_in3;
        __m256 prev_out0, prev_out1, prev_out2, prev_out3;
        prev_out3 = prev_out2 = prev_out1 = prev_out0 = _mm256_setzero_ps();
        prev_in3 = prev_in2 = prev_in1 = prev_in0 = _mm256_setzero_ps();

        float *tmpptr = tmp;

        // left border
        for (unsigned int i = coefs.n_border; i > 0; --i) {
            // load next eight pixels (one from each row)
            __m256 pixels = _mm256_set_ps(*(input + dim * n_pixels + i), *(input + (dim + 1) * n_pixels + i),
                                          *(input + (dim + 2) * n_pixels + i), *(input + (dim + 3) * n_pixels + i),
                                          *(input + (dim + 4) * n_pixels + i), *(input + (dim + 5) * n_pixels + i),
                                          *(input + (dim + 6) * n_pixels + i), *(input + (dim + 7) * n_pixels + i));

            // compute sum of products between ins/outs and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(pixels, mm_n_causal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in0, mm_n_causal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_causal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_causal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = pixels;
        }

        // causal pass
        for (unsigned int i = 0; i < n_pixels; ++i) {
            // load next eight pixels (one from each row)
            __m256 pixels = _mm256_set_ps(*(input + dim * n_pixels + i), *(input + (dim + 1) * n_pixels + i),
                                          *(input + (dim + 2) * n_pixels + i), *(input + (dim + 3) * n_pixels + i),
                                          *(input + (dim + 4) * n_pixels + i), *(input + (dim + 5) * n_pixels + i),
                                          *(input + (dim + 6) * n_pixels + i), *(input + (dim + 7) * n_pixels + i));

            // compute sum of products between ins/outs and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(pixels, mm_n_causal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in0, mm_n_causal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_causal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_causal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // store causal output in temporary buffer
            _mm256_store_ps(tmpptr, pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = pixels;

            tmpptr += 8;
        }

        // reset variables
        tmpptr -= 8;
        prev_out3 = prev_out2 = prev_out1 = prev_out0 = _mm256_setzero_ps();
        prev_in3 = prev_in2 = prev_in1 = prev_in0 = _mm256_setzero_ps();

        // right border
        for (unsigned int i = n_pixels - coefs.n_border; i < n_pixels; ++i) {
            // add products between pixels and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(prev_in0, mm_n_anticausal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_anticausal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_anticausal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in3, mm_n_anticausal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in3 = prev_in2;
            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = _mm256_set_ps(*(input + dim * n_pixels + i), *(input + (dim + 1) * n_pixels + i),
                                     *(input + (dim + 2) * n_pixels + i), *(input + (dim + 3) * n_pixels + i),
                                     *(input + (dim + 4) * n_pixels + i), *(input + (dim + 5) * n_pixels + i),
                                     *(input + (dim + 6) * n_pixels + i), *(input + (dim + 7) * n_pixels + i));
        }

        // anti-causal pass
        for (int i = n_pixels - 1; i >= 0; --i) {
            // add products between pixels and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(prev_in0, mm_n_anticausal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_anticausal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_anticausal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in3, mm_n_anticausal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // add output from causal pass
            __m256 pixels_res_causal = _mm256_load_ps(tmpptr);

            pixels_res_causal = _mm256_add_ps(pixels_res_causal, pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in3 = prev_in2;
            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = _mm256_set_ps(*(input + dim * n_pixels + i), *(input + (dim + 1) * n_pixels + i),
                                     *(input + (dim + 2) * n_pixels + i), *(input + (dim + 3) * n_pixels + i),
                                     *(input + (dim + 4) * n_pixels + i), *(input + (dim + 5) * n_pixels + i),
                                     *(input + (dim + 6) * n_pixels + i), *(input + (dim + 7) * n_pixels + i));

            // store outputs in correct row
            alignas(16) float out[8];
            _mm256_store_ps(out, pixels_res_causal);
            for (unsigned int j = 0; j < 8; ++j)
                *(output + (dim + j) * n_pixels + i) = out[7 - j];

            tmpptr -= 8;
        }
    }

    optimized_convolve_iir_inner_single_noavx(input + n_times_avx * n_pixels, n_pixels, n_times_normal,
                                              output + n_times_avx * n_pixels, coefs);

    detail::avx_free(tmp);
}

void convolve_iir_outer_single_avx(const float *input, const unsigned int n_pixels, const unsigned n_times,
                                   float *output, const Coefficients &coefs)
{
    __m256 mm_n_causal[4];
    __m256 mm_n_anticausal[4];
    __m256 mm_d[4];

    const unsigned n_times_avx = n_times & ~7;
    const unsigned n_times_normal = n_times - n_times_avx;

    float *tmp = (float *)detail::avx_memalign(sizeof(float) * n_pixels * 8);

    for (unsigned int i = 0; i < 4; ++i) {
        mm_n_causal[i] = _mm256_set1_ps(coefs.n_causal[i]);
        mm_n_anticausal[i] = _mm256_set1_ps(coefs.n_anticausal[i]);
        mm_d[i] = _mm256_set1_ps(-coefs.d[i]);
    }

    for (unsigned int dim = 0; dim < n_times_avx; dim += 8) {
        __m256 prev_in0, prev_in1, prev_in2, prev_in3;
        __m256 prev_out0, prev_out1, prev_out2, prev_out3;
        prev_out3 = prev_out2 = prev_out1 = prev_out0 = _mm256_setzero_ps();
        prev_in3 = prev_in2 = prev_in1 = prev_in0 = _mm256_setzero_ps();

        // left border
        for (unsigned int i = coefs.n_border; i > 0; --i) {
            __m256 pixels = _mm256_loadu_ps(input + dim + i * n_times);

            // compute sum of products between ins/outs and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(pixels, mm_n_causal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in0, mm_n_causal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_causal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_causal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = pixels;
        }

        // causal pass
        for (unsigned int i = 0; i < n_pixels; ++i) {
            // load next eight pixels (one from each row)
            __m256 pixels = _mm256_loadu_ps(input + dim + i * n_times);

            // compute sum of products between ins/outs and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(pixels, mm_n_causal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in0, mm_n_causal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_causal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_causal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // store causal output in temporary buffer
            _mm256_store_ps(tmp + i * 8, pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = pixels;
        }

        // reset variables
        prev_out3 = prev_out2 = prev_out1 = prev_out0 = _mm256_setzero_ps();
        prev_in3 = prev_in2 = prev_in1 = prev_in0 = _mm256_setzero_ps();

        // right border
        for (unsigned int i = n_pixels - 1 - coefs.n_border; i < n_pixels; ++i) {
            // add products between pixels and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(prev_in0, mm_n_anticausal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_anticausal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_anticausal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in3, mm_n_anticausal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in3 = prev_in2;
            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = _mm256_loadu_ps(input + dim + i * n_times);
        }

        // anti-causal pass
        for (int i = n_pixels - 1; i >= 0; --i) {
            // add products between pixels and kernel coefficients
            __m256 pixels_res = _mm256_mul_ps(prev_in0, mm_n_anticausal[0]);
            pixels_res = _mm256_fmadd_ps(prev_in1, mm_n_anticausal[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in2, mm_n_anticausal[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_in3, mm_n_anticausal[3], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out0, mm_d[0], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out1, mm_d[1], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out2, mm_d[2], pixels_res);
            pixels_res = _mm256_fmadd_ps(prev_out3, mm_d[3], pixels_res);

            // add output from causal pass
            __m256 pixels_res_causal = _mm256_load_ps(tmp + i * 8);
            pixels_res_causal = _mm256_add_ps(pixels_res_causal, pixels_res);

            // move to next pixel
            prev_out3 = prev_out2;
            prev_out2 = prev_out1;
            prev_out1 = prev_out0;
            prev_out0 = pixels_res;

            prev_in3 = prev_in2;
            prev_in2 = prev_in1;
            prev_in1 = prev_in0;
            prev_in0 = _mm256_loadu_ps(input + dim + i * n_times);

            // store outputs in correct row
            _mm256_storeu_ps(output + dim + i * n_times, pixels_res_causal);
        }
    }

    optimized_convolve_iir_outer_single_noavx(input + n_times_avx, n_pixels, n_times_normal, output + n_times_avx,
                                              coefs, n_times);

    detail::avx_free(tmp);
}

} // namespace detail

} // namespace fastfilters
