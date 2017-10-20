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
#include "avx_mathfun.h"

#include <immintrin.h>

static inline void swap(float *a, float *b)
{
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

static inline float max(float a, float b)
{
    if (a > b)
        return a;
    else
        return b;
}


#ifdef __AVX2__
#define fname _ev3d_avx2
#elif defined(__AVX__)
#define fname _ev3d_avx
#else
#error "linalg_avx2.c needs to be compiled with avx or avx2 support"
#endif

static inline __m256 _mm256_abs_ps(__m256 x)
{
    __m256 m = _mm256_set1_ps(-0.f);
    return _mm256_andnot_ps(m, x);
}


DLL_LOCAL void fname(const float *a00, const float *a01, const float *a02, const float *a11, const float *a12,
                     const float *a22, float *ev0, float *ev1, float *ev2, const size_t len)
{
    const size_t avx_end = len & ~7;
    __m256 v_inv3 = _mm256_set1_ps(1.0 / 3.0);
    __m256 v_root3 = _mm256_sqrt_ps(_mm256_set1_ps(3.0));
    __m256 two = _mm256_set1_ps(2.0);
    __m256 half = _mm256_set1_ps(0.5);
    __m256 zero = _mm256_setzero_ps();

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 v_a00 = _mm256_loadu_ps(a00 + i);
        __m256 v_a01 = _mm256_loadu_ps(a01 + i);
        __m256 v_a02 = _mm256_loadu_ps(a02 + i);
        __m256 v_a11 = _mm256_loadu_ps(a11 + i);
        __m256 v_a12 = _mm256_loadu_ps(a12 + i);
        __m256 v_a22 = _mm256_loadu_ps(a22 + i);

        // guard against float overflows
        __m256 v_max0 = _mm256_max_ps(_mm256_abs_ps(v_a00), _mm256_abs_ps(v_a01));
        __m256 v_max1 = _mm256_max_ps(_mm256_abs_ps(v_a02), _mm256_abs_ps(v_a11));
        __m256 v_max2 = _mm256_max_ps(_mm256_abs_ps(v_a12), _mm256_abs_ps(v_a22));
        __m256 v_max_element = _mm256_max_ps(_mm256_max_ps(v_max0, v_max1), v_max2);

        v_a00 = _mm256_div_ps(v_a00, v_max_element);
        v_a01 = _mm256_div_ps(v_a01, v_max_element);
        v_a02 = _mm256_div_ps(v_a02, v_max_element);
        v_a11 = _mm256_div_ps(v_a11, v_max_element);
        v_a12 = _mm256_div_ps(v_a12, v_max_element);
        v_a22 = _mm256_div_ps(v_a22, v_max_element);

        __m256 c0 = _avx_sub(_avx_sub(_avx_sub(_avx_add(_avx_mul(_avx_mul(v_a00, v_a11), v_a22),
            _avx_mul(_avx_mul(_avx_mul(two, v_a01), v_a02), v_a12)),
            _avx_mul(_avx_mul(v_a00, v_a12), v_a12)),
            _avx_mul(_avx_mul(v_a11, v_a02), v_a02)),
            _avx_mul(_avx_mul(v_a22, v_a01), v_a01));
        __m256 c1 = _avx_sub(_avx_add(_avx_sub(_avx_add(_avx_sub(_avx_mul(v_a00, v_a11),
            _avx_mul(v_a01, v_a01)),
            _avx_mul(v_a00, v_a22)),
            _avx_mul(v_a02, v_a02)),
            _avx_mul(v_a11, v_a22)),
            _avx_mul(v_a12, v_a12));
        __m256 c2 = _avx_add(_avx_add(v_a00, v_a11), v_a22);
        __m256 c2Div3 = _avx_mul(c2, v_inv3);
        __m256 aDiv3 = _avx_mul(_avx_sub(c1, _avx_mul(c2, c2Div3)), v_inv3);

        aDiv3 = _mm256_min_ps(aDiv3, zero);

        __m256 mbDiv2 = _avx_mul(half, _avx_add(c0, _avx_mul(c2Div3, _avx_sub(_avx_mul(_avx_mul(two, c2Div3), c2Div3), c1))));
        __m256 q = _avx_add(_avx_mul(mbDiv2, mbDiv2), _avx_mul(_avx_mul(aDiv3, aDiv3), aDiv3));

        q = _mm256_min_ps(q, zero);

        __m256 magnitude = _mm256_sqrt_ps(_avx_neg(aDiv3));
        __m256 angle = _avx_mul(atan2_256_ps(_mm256_sqrt_ps(_avx_neg(q)), mbDiv2), v_inv3);
        __m256 cs, sn;

        sincos256_ps(angle, &sn, &cs);

        __m256 r0 = _avx_add(c2Div3, _avx_mul(_avx_mul(two, magnitude), cs));
        __m256 r1 = _avx_sub(c2Div3, _avx_mul(magnitude, _avx_add(cs, _avx_mul(v_root3, sn))));
        __m256 r2 = _avx_sub(c2Div3, _avx_mul(magnitude, _avx_sub(cs, _avx_mul(v_root3, sn))));

        __m256 v_r0_tmp = _mm256_min_ps(r0, r1);
        __m256 v_r1_tmp = _mm256_max_ps(r0, r1);

        __m256 v_r0 = _mm256_min_ps(v_r0_tmp, r2);
        __m256 v_r2_tmp = _mm256_max_ps(v_r0_tmp, r2);

        __m256 v_r1 = _mm256_min_ps(v_r1_tmp, v_r2_tmp);
        __m256 v_r2 = _mm256_max_ps(v_r1_tmp, v_r2_tmp);

        v_r0 = _mm256_mul_ps(v_r0, v_max_element);
        v_r1 = _mm256_mul_ps(v_r1, v_max_element);
        v_r2 = _mm256_mul_ps(v_r2, v_max_element);

        _mm256_storeu_ps(ev2 + i, v_r0);
        _mm256_storeu_ps(ev1 + i, v_r1);
        _mm256_storeu_ps(ev0 + i, v_r2);
    }

    const float inv3 = 1.0 / 3.0;
    const float root3 = sqrt(3.0);
    for (size_t i = avx_end; i < len; ++i) {
        float i_a00 = a00[i];
        float i_a01 = a01[i];
        float i_a02 = a02[i];
        float i_a11 = a11[i];
        float i_a12 = a12[i];
        float i_a22 = a22[i];

        // guard against float overflows
        float max0 = max(fabs(i_a00), fabs(i_a01));
        float max1 = max(fabs(i_a02), fabs(i_a11));
        float max2 = max(fabs(i_a12), fabs(i_a22));
        float maxElement = max(max(max0, max1), max2);
        float invMaxElement = 1/maxElement;

        i_a00 *= invMaxElement;
        i_a01 *= invMaxElement;
        i_a02 *= invMaxElement;
        i_a11 *= invMaxElement;
        i_a12 *= invMaxElement;
        i_a22 *= invMaxElement;

        float c0 = i_a00 * i_a11 * i_a22 + 2.0 * i_a01 * i_a02 * i_a12 - i_a00 * i_a12 * i_a12 -
                   i_a11 * i_a02 * i_a02 - i_a22 * i_a01 * i_a01;
        float c1 =
            i_a00 * i_a11 - i_a01 * i_a01 + i_a00 * i_a22 - i_a02 * i_a02 + i_a11 * i_a22 - i_a12 * i_a12;
        float c2 = i_a00 + i_a11 + i_a22;
        float c2Div3 = c2 * inv3;
        float aDiv3 = (c1 - c2 * c2Div3) * inv3;

        if (aDiv3 > 0.0)
            aDiv3 = 0.0;

        float mbDiv2 = 0.5 * (c0 + c2Div3 * (2.0 * c2Div3 * c2Div3 - c1));
        float q = mbDiv2 * mbDiv2 + aDiv3 * aDiv3 * aDiv3;

        if (q > 0.0)
            q = 0.0;

        float magnitude = sqrt(-aDiv3);
        float angle = atan2(sqrt(-q), mbDiv2) * inv3;
        float cs = cos(angle);
        float sn = sin(angle);
        float r0 = (c2Div3 + 2.0 * magnitude * cs);
        float r1 = (c2Div3 - magnitude * (cs + root3 * sn));
        float r2 = (c2Div3 - magnitude * (cs - root3 * sn));

        if (r0 < r1)
            swap(&r0, &r1);
        if (r0 < r2)
            swap(&r0, &r2);
        if (r1 < r2)
            swap(&r1, &r2);

        ev0[i] = r0;
        ev1[i] = r1;
        ev2[i] = r2;
    }
}