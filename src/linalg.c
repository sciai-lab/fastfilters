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

typedef void (*ev2d_fn_t)(const float *, const float *, const float *, float *, float *, const size_t);
typedef void (*combine_add_fn_t)(const float *, const float *, float *, size_t);

void DLL_LOCAL _ev2d_avx(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big,
                         const size_t len);

void DLL_LOCAL _combine_add_avx(const float *a, const float *b, float *c, size_t len);
void DLL_LOCAL _combine_addsqrt_avx(const float *a, const float *b, float *c, size_t len);

static void _ev2d_default(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big,
                          const size_t len)
{
    for (size_t i = 0; i < len; i++) {
        float v_xx = xx[i];
        float v_xy = xy[i];
        float v_yy = yy[i];

        float T = v_xx + v_yy;
        float Thalf = T / 2;
        float Thalfsq = Thalf * Thalf;

        float D = v_xx * v_yy + v_xy * v_xy;

        float Dsqrt = sqrt(Thalfsq - D);

        float ev0 = Thalf + Dsqrt;
        float ev1 = Thalf - Dsqrt;

        if (ev0 > ev1) {
            ev_small[i] = ev1;
            ev_big[i] = ev0;
        } else {
            ev_small[i] = ev0;
            ev_big[i] = ev1;
        }
    }
}

static void _combine_add_default(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

static void _combine_addsqrt_default(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = sqrt(a[i] + b[i]);
}

static ev2d_fn_t g_ev2d_fn = NULL;
static combine_add_fn_t g_combine_add = NULL;
static combine_add_fn_t g_combine_addsqrt = NULL;

void fastfilters_linalg_init()
{
    if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX))
        g_ev2d_fn = _ev2d_avx;
    else
        g_ev2d_fn = _ev2d_default;

    if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX))
        g_combine_add = _combine_add_avx;
    else
        g_combine_add = _combine_add_default;

    if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX))
        g_combine_addsqrt = _combine_addsqrt_avx;
    else
        g_combine_addsqrt = _combine_addsqrt_default;
}

void DLL_PUBLIC fastfilters_linalg_ev2d(const float *xx, const float *xy, const float *yy, float *ev_small,
                                        float *ev_big, const size_t len)
{
    g_ev2d_fn(xx, xy, yy, ev_small, ev_big, len);
}

void DLL_PUBLIC fastfilters_combine_add2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                          fastfilters_array2d_t *out)
{
    g_combine_add(a->ptr, b->ptr, out->ptr, a->n_y * a->stride_y);
}

void DLL_PUBLIC fastfilters_combine_addsqrt2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                              fastfilters_array2d_t *out)
{
    g_combine_addsqrt(a->ptr, b->ptr, out->ptr, a->n_y * a->stride_y);
}