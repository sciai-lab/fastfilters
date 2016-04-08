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

#include "fastfilters.h"
#include "common.h"

#include <immintrin.h>

#ifndef __AVX__
#error "linalg_avx.c need to be compiled with AVX support"
#endif

void _ev2d_avx(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big, const size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 v_xx, v_xy, v_yy;

        v_xx = _mm256_loadu_ps(xx + i);
        v_xy = _mm256_loadu_ps(xy + i);
        v_yy = _mm256_loadu_ps(yy + i);

        __m256 thalf, thalfsq;
        thalf = _mm256_mul_ps(_mm256_add_ps(v_xx, v_yy), _mm256_set1_ps(0.5));
        thalfsq = _mm256_mul_ps(thalf, thalf);

        __m256 d = _mm256_sub_ps(_mm256_mul_ps(v_xx, v_yy), _mm256_mul_ps(v_xy, v_xy));

        __m256 det = _mm256_sqrt_ps(_mm256_add_ps(thalfsq, d));

        __m256 ev0 = _mm256_add_ps(thalf, det);
        __m256 ev1 = _mm256_sub_ps(thalf, det);

        __m256 mask = _mm256_cmp_ps(ev0, ev1, _CMP_LE_OQ);

        __m256 v_ev_small = _mm256_or_ps(_mm256_and_ps(mask, ev1), _mm256_andnot_ps(mask, ev0));
        __m256 v_ev_big = _mm256_or_ps(_mm256_andnot_ps(mask, ev1), _mm256_and_ps(mask, ev0));

        _mm256_storeu_ps(ev_small + i, v_ev_small);
        _mm256_storeu_ps(ev_big + i, v_ev_big);
    }
}