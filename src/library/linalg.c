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
typedef void (*ev3d_fn_t)(const float *, const float *, const float *, const float *, const float *, const float *,
                          float *, float *, float *, const size_t);

typedef void (*combine_add_fn_t)(const float *, const float *, float *, size_t);
typedef void (*combine_add3_fn_t)(const float *, const float *, const float *, float *, size_t);

void DLL_LOCAL _ev2d_avx(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big,
                         const size_t len);

void DLL_LOCAL _combine_add_avx(const float *a, const float *b, float *c, size_t len);
void DLL_LOCAL _combine_addsqrt_avx(const float *a, const float *b, float *c, size_t len);
void DLL_LOCAL _combine_mul_avx(const float *a, const float *b, float *c, size_t len);

void DLL_LOCAL _combine_add3_avx(const float *a, const float *b, const float *c, float *res, size_t len);
void DLL_LOCAL _combine_addsqrt3_avx(const float *a, const float *b, const float *c, float *res, size_t len);

DLL_LOCAL void _ev3d_avx(const float *a00, const float *a01, const float *a02, const float *a11, const float *a12,
                         const float *a22, float *ev0, float *ev1, float *ev2, const size_t len);
DLL_LOCAL void _ev3d_avx2(const float *a00, const float *a01, const float *a02, const float *a11, const float *a12,
                          const float *a22, float *ev0, float *ev1, float *ev2, const size_t len);

static void _ev2d_default(const float *xx, const float *xy, const float *yy, float *ev_big, float *ev_small,
                          const size_t len)
{
    for (size_t i = 0; i < len; ++i) {
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

/*
based on vigra's include/vigra/mathutil.hxx with the following license:


     \brief Compute the eigenvalues of a 3x3 real symmetric matrix.
        This uses a numerically stable version of the analytical eigenvalue formula according to
        <p>
        David Eberly: <a href="http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf">
        <em>"Eigensystems for 3 x 3 Symmetric Matrices (Revisited)"</em></a>, Geometric Tools Documentation, 2006
        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
*/
/************************************************************************/
/*                                                                      */
/*               Copyright 1998-2011 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

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

static void _ev3d_default(const float *a00, const float *a01, const float *a02, const float *a11, const float *a12,
                          const float *a22, float *ev0, float *ev1, float *ev2, const size_t len)
{
    const float inv3 = 1.0 / 3.0;
    const float root3 = sqrt(3.0);

    for (size_t i = 0; i < len; ++i) {
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

        if (maxElement == 0) {
            ev0[i] = 0;
            ev1[i] = 0;
            ev2[i] = 0;
            continue;
        }

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

        ev0[i] = r0 * maxElement;
        ev1[i] = r1 * maxElement;
        ev2[i] = r2 * maxElement;
    }
}

static void _combine_add_default(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] + b[i];
}

static void _combine_add3_default(const float *a, const float *b, const float *c, float *res, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = a[i] + b[i] + c[i];
}

static void _combine_mul_default(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = a[i] * b[i];
}

static void _combine_addsqrt_default(const float *a, const float *b, float *c, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        c[i] = sqrt(a[i] * a[i] + b[i] * b[i]);
}

static void _combine_addsqrt3_default(const float *a, const float *b, const float *c, float *res, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        res[i] = sqrt(a[i] * a[i] + b[i] * b[i] + c[i] * c[i]);
}

static ev2d_fn_t g_ev2d_fn = NULL;
static ev3d_fn_t g_ev3d_fn = NULL;
static combine_add_fn_t g_combine_add = NULL;
static combine_add_fn_t g_combine_mul = NULL;
static combine_add_fn_t g_combine_addsqrt = NULL;
static combine_add3_fn_t g_combine_add3 = NULL;
static combine_add3_fn_t g_combine_addsqrt3 = NULL;

void fastfilters_linalg_init()
{
    if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX)) {
        g_combine_add = _combine_add_avx;
        g_combine_add3 = _combine_add3_avx;
        g_combine_mul = _combine_mul_avx;
        g_combine_addsqrt = _combine_addsqrt_avx;
        g_combine_addsqrt3 = _combine_addsqrt3_avx;
        g_ev2d_fn = _ev2d_avx;
    } else {
        g_combine_add = _combine_add_default;
        g_combine_add3 = _combine_add3_default;
        g_combine_mul = _combine_mul_default;
        g_combine_addsqrt = _combine_addsqrt_default;
        g_combine_addsqrt3 = _combine_addsqrt3_default;
        g_ev2d_fn = _ev2d_default;
    }

    if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX2)) {
        g_ev3d_fn = _ev3d_avx2;
    } else if (fastfilters_cpu_check(FASTFILTERS_CPU_AVX)) {
        g_ev3d_fn = _ev3d_avx;
    } else {
        g_ev3d_fn = _ev3d_default;
    }
}

void DLL_PUBLIC fastfilters_linalg_ev3d(const float *a00, const float *a01, const float *a02, const float *a11,
                                        const float *a12, const float *a22, float *ev0, float *ev1, float *ev2,
                                        const size_t len)
{
    g_ev3d_fn(a00, a01, a02, a11, a12, a22, ev0, ev1, ev2, len);
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

void DLL_PUBLIC fastfilters_combine_mul2d(const fastfilters_array2d_t *a, const fastfilters_array2d_t *b,
                                          fastfilters_array2d_t *out)
{
    g_combine_mul(a->ptr, b->ptr, out->ptr, a->n_y * a->stride_y);
}

void DLL_PUBLIC fastfilters_combine_mul3d(const fastfilters_array3d_t *a, const fastfilters_array3d_t *b,
                                          fastfilters_array3d_t *out)
{
    g_combine_mul(a->ptr, b->ptr, out->ptr, a->n_z * a->stride_z);
}

void DLL_PUBLIC fastfilters_combine_add3d(const fastfilters_array3d_t *a, const fastfilters_array3d_t *b,
                                          const fastfilters_array3d_t *c, fastfilters_array3d_t *out)
{
    g_combine_add3(a->ptr, b->ptr, c->ptr, out->ptr, a->n_z * a->stride_z);
}

void DLL_PUBLIC fastfilters_combine_addsqrt3d(const fastfilters_array3d_t *a, const fastfilters_array3d_t *b,
                                              const fastfilters_array3d_t *c, fastfilters_array3d_t *out)
{
    g_combine_addsqrt3(a->ptr, b->ptr, c->ptr, out->ptr, a->n_z * a->stride_z);
}
