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

void DLL_LOCAL _ev2d_avx(const float *xx, const float *xy, const float *yy, float *ev_big, float *ev_small,
                         const size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 v_xx, v_xy, v_yy;

        v_xx = _mm256_loadu_ps(xx + i);
        v_xy = _mm256_loadu_ps(xy + i);
        v_yy = _mm256_loadu_ps(yy + i);

        __m256 tmp0 = _mm256_mul_ps(_mm256_add_ps(v_xx, v_yy), _mm256_set1_ps(0.5));
        __m256 tmp1 = _mm256_mul_ps(_mm256_sub_ps(v_xx, v_yy), _mm256_set1_ps(0.5));
        tmp1 = _mm256_mul_ps(tmp1, tmp1);

        __m256 det = _mm256_sqrt_ps(_mm256_add_ps(tmp1, _mm256_mul_ps(v_xy, v_xy)));

        __m256 ev0 = _mm256_add_ps(tmp0, det);
        __m256 ev1 = _mm256_sub_ps(tmp0, det);

        __m256 mask = _mm256_cmp_ps(ev0, ev1, _CMP_LE_OQ);

        __m256 v_ev_big = _mm256_or_ps(_mm256_and_ps(mask, ev1), _mm256_andnot_ps(mask, ev0));
        __m256 v_ev_small = _mm256_or_ps(_mm256_andnot_ps(mask, ev1), _mm256_and_ps(mask, ev0));

        _mm256_storeu_ps(ev_small + i, v_ev_small);
        _mm256_storeu_ps(ev_big + i, v_ev_big);
    }

    for (size_t i = avx_end; i < len; i++) {
        float v_xx = xx[i];
        float v_xy = xy[i];
        float v_yy = yy[i];

        float tmp0 = (v_xx + v_yy) / 2.0;

        float tmp1 = (v_xx - v_yy) / 2.0;
        tmp1 = tmp1 * tmp1;

        float det = (tmp1 + v_xy * v_xy);
        float det_sqrt = sqrt(det);

        float ev0 = tmp0 + det_sqrt;
        float ev1 = tmp0 - det_sqrt;

        if (ev0 > ev1) {
            ev_small[i] = ev1;
            ev_big[i] = ev0;
        } else {
            ev_small[i] = ev0;
            ev_big[i] = ev1;
        }
    }
}

void DLL_LOCAL _combine_add_avx(const float *a, const float *b, float *c, size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 va, vb;
        va = _mm256_loadu_ps(a + i);
        vb = _mm256_loadu_ps(b + i);

        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }

    for (size_t i = avx_end; i < len; i++)
        c[i] = a[i] + b[i];
}

void DLL_LOCAL _combine_add3_avx(const float *a, const float *b, const float *c, float *res, size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 va, vb, vc;
        va = _mm256_loadu_ps(a + i);
        vb = _mm256_loadu_ps(b + i);
        vc = _mm256_loadu_ps(c + i);

        _mm256_storeu_ps(res + i, _mm256_add_ps(_mm256_add_ps(va, vb), vc));
    }

    for (size_t i = avx_end; i < len; i++)
        res[i] = a[i] + b[i] + c[i];
}

void DLL_LOCAL _combine_addsqrt_avx(const float *a, const float *b, float *c, size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 va, vb;
        va = _mm256_loadu_ps(a + i);
        vb = _mm256_loadu_ps(b + i);

        va = _mm256_mul_ps(va, va);
        vb = _mm256_mul_ps(vb, vb);

        _mm256_storeu_ps(c + i, _mm256_sqrt_ps(_mm256_add_ps(va, vb)));
    }

    for (size_t i = avx_end; i < len; i++)
        c[i] = sqrt(a[i] * a[i] + b[i] * b[i]);
}

// Alefeld, G. "On the convergence of Halley's Method." The American Mathematical Monthly 88.7 (1981): 530-536.
extern inline __m256 mm256_cbrt_ps(__m256 q)
{
    __m256 two = _mm256_set1_ps(2.0);
    __m256 twoq = _mm256_mul_ps(two, q);

    __m256 c0 = _mm256_set1_ps(1.4774329094);
    __m256 c1 = _mm256_set1_ps(0.8414323527);
    __m256 c2 = _mm256_set1_ps(0.7387320679);
    __m256 xn = _mm256_sub_ps(c0, _mm256_div_ps(c1, _mm256_add_ps(c2, q)));

    for (unsigned int i = 0; i < 20; ++i) {
        __m256 nom, denom;
        __m256 xn_cubed = _mm256_mul_ps(xn, _mm256_mul_ps(xn, xn));

        nom = _mm256_mul_ps(xn, _mm256_add_ps(xn_cubed, twoq));
        denom = _mm256_add_ps(_mm256_mul_ps(two, xn_cubed), q);

        xn = _mm256_div_ps(nom, denom);
    }

    return xn;
}

void DLL_LOCAL _combine_addsqrt3_avx(const float *a, const float *b, const float *c, float *res, size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 va, vb, vc;
        va = _mm256_loadu_ps(a + i);
        vb = _mm256_loadu_ps(b + i);
        vc = _mm256_loadu_ps(c + i);

        va = _mm256_mul_ps(va, va);
        vb = _mm256_mul_ps(vb, vb);
        vc = _mm256_mul_ps(vc, vc);

        __m256 sum = _mm256_add_ps(_mm256_add_ps(va, vb), vc);

        _mm256_storeu_ps(res + i, mm256_cbrt_ps(sum));
    }

    for (size_t i = avx_end; i < len; i++)
        res[i] = cbrt(a[i] * a[i] + b[i] * b[i] + c[i] * c[i]);
}

void DLL_LOCAL _combine_mul_avx(const float *a, const float *b, float *c, size_t len)
{
    const size_t avx_end = len & ~7;

    for (size_t i = 0; i < avx_end; i += 8) {
        __m256 va, vb;
        va = _mm256_loadu_ps(a + i);
        vb = _mm256_loadu_ps(b + i);

        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
    }

    for (size_t i = avx_end; i < len; i++)
        c[i] = a[i] * b[i];
}