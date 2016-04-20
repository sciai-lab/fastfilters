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

#ifndef FASTFILTERS_FIR_CONVOLVE_NOSIMD_IMPL_H
#error "Do not include/compile fir_convolve_nosimd_impl.h directly"
#endif

#if !defined(FF_BOUNDARY_OPTIMISTIC_LEFT) && !defined(FF_BOUNDARY_MIRROR_LEFT) && !defined(FF_BOUNDARY_PTR_LEFT)

#define FF_BOUNDARY_OPTIMISTIC_LEFT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_OPTIMISTIC_LEFT

#define FF_BOUNDARY_MIRROR_LEFT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_MIRROR_LEFT

#define FF_BOUNDARY_PTR_LEFT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_PTR_LEFT

#elif !defined(FF_BOUNDARY_OPTIMISTIC_RIGHT) && !defined(FF_BOUNDARY_MIRROR_RIGHT) && !defined(FF_BOUNDARY_PTR_RIGHT)

#define FF_BOUNDARY_OPTIMISTIC_RIGHT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_OPTIMISTIC_RIGHT

#define FF_BOUNDARY_MIRROR_RIGHT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_MIRROR_RIGHT

#define FF_BOUNDARY_PTR_RIGHT
#include "fir_convolve_nosimd_impl.h"
#undef FF_BOUNDARY_PTR_RIGHT

#elif !defined(FF_KERNEL_SYMMETRIC) && !defined(FF_KERNEL_ANTISYMMETRIC)

#define FF_KERNEL_SYMMETRIC
#include "fir_convolve_nosimd_impl.h"
#undef FF_KERNEL_SYMMETRIC

#define FF_KERNEL_ANTISYMMETRIC
#include "fir_convolve_nosimd_impl.h"
#undef FF_KERNEL_ANTISYMMETRIC

#else

#ifdef FF_BOUNDARY_OPTIMISTIC_LEFT
#define boundary_name_left optimistic_
#elif defined(FF_BOUNDARY_MIRROR_LEFT)
#define boundary_name_left mirror_
#elif defined(FF_BOUNDARY_PTR_LEFT)
#define boundary_name_left ptr_
#else
#error "No boundary treatment mode defined."
#endif

#ifdef FF_BOUNDARY_OPTIMISTIC_RIGHT
#define boundary_name_right optimistic_
#elif defined(FF_BOUNDARY_MIRROR_RIGHT)
#define boundary_name_right mirror_
#elif defined(FF_BOUNDARY_PTR_RIGHT)
#define boundary_name_right ptr_
#else
#error "No boundary treatment mode defined."
#endif

#ifdef FF_KERNEL_SYMMETRIC
#define symmetry_name symmetric
#elif defined(FF_KERNEL_ANTISYMMETRIC)
#define symmetry_name antisymmetric
#else
#error "FF_KERNEL_SYMMETRIC and FF_KERNEL_ANTISYMMETRIC not defined"
#endif

#define boundary_name BOOST_PP_CAT(boundary_name_left, boundary_name_right)

#ifdef KERNEL_LEN_RUNTIME
#define KERNEL_LEN kernel->len
#define FNAME BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_impl_, BOOST_PP_CAT(boundary_name, symmetry_name)), N)
#else
#define KERNEL_LEN BOOST_PP_ITERATION()
#define FNAME                                                                                                          \
    BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_impl_, BOOST_PP_CAT(boundary_name, symmetry_name)), BOOST_PP_ITERATION())
#endif

static bool FNAME(const float *inptr, const float *in_border_left, const float *in_border_right, size_t n_pixels,
                  size_t pixel_stride, size_t n_outer, size_t outer_stride, float *outptr, size_t outptr_outer_stride,
                  size_t borderptr_outer_stride, const fastfilters_kernel_fir_t kernel)

{
#ifndef FF_BOUNDARY_PTR_RIGHT
    (void)in_border_right;
#endif
#ifndef FF_BOUNDARY_PTR_LEFT
    (void)in_border_left;
#endif
#if !defined(FF_BOUNDARY_PTR_LEFT) && !defined(FF_BOUNDARY_PTR_RIGHT)
    (void)borderptr_outer_stride;
#endif

    for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
        const float *cur_inptr = inptr + outer_stride * i_outer;
        float *cur_outptr = outptr + outptr_outer_stride * i_outer;

        unsigned int i_inner = 0;

// left border
#ifdef FF_BOUNDARY_MIRROR_LEFT
        for (unsigned int j = 0; j < KERNEL_LEN; ++j) {
            float sum = 0.0;

            sum = kernel->coefs[0] * cur_inptr[j * pixel_stride];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)j < 0)
                    offset_left = -j + k;
                else
                    offset_left = j - k;

                if (k + j >= n_pixels)
                    offset_right = n_pixels - ((k + j) % n_pixels) - 2;
                else
                    offset_right = j + k;
#ifdef FF_KERNEL_SYMMETRIC
                sum +=
                    kernel->coefs[k] * (cur_inptr[offset_right * pixel_stride] + cur_inptr[offset_left * pixel_stride]);
#else
                sum +=
                    kernel->coefs[k] * (cur_inptr[offset_right * pixel_stride] - cur_inptr[offset_left * pixel_stride]);
#endif
            }

            cur_outptr[j * pixel_stride] = sum;
        }

        i_inner = KERNEL_LEN;
#endif

#ifdef FF_BOUNDARY_PTR_LEFT
        for (unsigned int j = 0; j < KERNEL_LEN; ++j) {
            float sum = 0.0;

            sum = kernel->coefs[0] * cur_inptr[j * pixel_stride];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                float left;
                if (-(int)k + (int)j < 0)
                    left = in_border_left[i_outer * borderptr_outer_stride +
                                          (KERNEL_LEN - (int)k + (int)j) * pixel_stride];
                else
                    left = cur_inptr[(j - k) * pixel_stride];

#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_inptr[(j + k) * pixel_stride] + left);
#else
                sum += kernel->coefs[k] * (cur_inptr[(j + k) * pixel_stride] - left);
#endif
            }

            cur_outptr[j * pixel_stride] = sum;
        }

        i_inner = KERNEL_LEN;
#endif

// 'valid' area of line
#if defined(FF_BOUNDARY_MIRROR_RIGHT) || defined(FF_BOUNDARY_PTR_RIGHT)
        const unsigned int end = n_pixels - KERNEL_LEN;
#else
        const unsigned int end = n_pixels;
#endif
        for (; i_inner < end; ++i_inner) {
            float sum = kernel->coefs[0] * cur_inptr[i_inner * pixel_stride];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                int offset_left = (int)(i_inner - k) * pixel_stride;
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (cur_inptr[(i_inner + k) * pixel_stride] + cur_inptr[offset_left]);
#else
                sum += kernel->coefs[k] * (cur_inptr[(i_inner + k) * pixel_stride] - cur_inptr[offset_left]);
#endif
            }

            cur_outptr[i_inner * pixel_stride] = sum;
        }

// right border
#ifdef FF_BOUNDARY_MIRROR_RIGHT
        for (; i_inner < n_pixels; ++i_inner) {
            float sum = 0.0;

            sum = kernel->coefs[0] * cur_inptr[i_inner * pixel_stride];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)i_inner < 0)
                    offset_left = -i_inner + k;
                else
                    offset_left = i_inner - k;

                if (k + i_inner >= n_pixels)
                    offset_right = n_pixels - ((k + i_inner) % n_pixels) - 2;
                else
                    offset_right = i_inner + k;
#ifdef FF_KERNEL_SYMMETRIC
                sum +=
                    kernel->coefs[k] * (cur_inptr[offset_right * pixel_stride] + cur_inptr[offset_left * pixel_stride]);
#else
                sum +=
                    kernel->coefs[k] * (cur_inptr[offset_right * pixel_stride] - cur_inptr[offset_left * pixel_stride]);
#endif
            }

            cur_outptr[i_inner * pixel_stride] = sum;
        }
#endif

#ifdef FF_BOUNDARY_PTR_RIGHT
        for (; i_inner < n_pixels; ++i_inner) {
            float sum = 0.0;

            sum = kernel->coefs[0] * cur_inptr[i_inner * pixel_stride];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                float right;

                if (k + i_inner >= n_pixels)
                    right = in_border_right[i_outer * borderptr_outer_stride + ((k + i_inner) % n_pixels)];
                else
                    right = cur_inptr[(i_inner + k) * pixel_stride];
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (right + cur_inptr[(i_inner - k) * pixel_stride]);
#else
                sum += kernel->coefs[k] * (right - cur_inptr[(i_inner - k) * pixel_stride]);
#endif
            }

            cur_outptr[i_inner * pixel_stride] = sum;
        }
#endif
    }

    return true;
}

#undef FNAME

#ifdef KERNEL_LEN_RUNTIME
#define FNAME BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_outer_impl_, BOOST_PP_CAT(boundary_name, symmetry_name)), N)
#else
#define FNAME                                                                                                          \
    BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_outer_impl_, BOOST_PP_CAT(boundary_name, symmetry_name)),                   \
                 BOOST_PP_ITERATION())
#endif

static bool FNAME(const float *inptr, const float *in_border_left, const float *in_border_right, size_t n_pixels,
                  size_t pixel_stride, size_t n_outer, size_t outer_stride, float *outptr, size_t outptr_outer_stride,
                  size_t borderptr_outer_stride, const fastfilters_kernel_fir_t kernel)
{
#ifndef FF_BOUNDARY_PTR_RIGHT
    (void)in_border_right;
#endif
#ifndef FF_BOUNDARY_PTR_LEFT
    (void)in_border_left;
#endif
#if !defined(FF_BOUNDARY_PTR_LEFT) && !defined(FF_BOUNDARY_PTR_RIGHT)
    (void)borderptr_outer_stride;
#endif

    if (n_outer != outptr_outer_stride)
        return false;

    float *tmp = fastfilters_memory_alloc((KERNEL_LEN + 1) * n_outer * sizeof(float));

    if (!tmp)
        return false;

    unsigned int i_pixel = 0;

// left border
#ifdef FF_BOUNDARY_MIRROR_LEFT
    for (; i_pixel < KERNEL_LEN; ++i_pixel) {
        for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
            float sum = 0.0;

            sum = kernel->coefs[0] * inptr[pixel_stride * i_pixel + outer_stride * i_outer];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)i_pixel < 0)
                    offset_left = -i_pixel + k;
                else
                    offset_left = i_pixel - k;

                if (k + i_pixel >= n_pixels)
                    offset_right = n_pixels - ((k + i_pixel) % n_pixels) - 2;
                else
                    offset_right = i_pixel + k;

#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (inptr[offset_right * pixel_stride + outer_stride * i_outer] +
                                           inptr[offset_left * pixel_stride + outer_stride * i_outer]);
#else
                sum += kernel->coefs[k] * (inptr[offset_right * pixel_stride + outer_stride * i_outer] -
                                           inptr[offset_left * pixel_stride + outer_stride * i_outer]);
#endif
            }

            tmp[n_outer * i_pixel + i_outer] = sum;
        }
    }
#endif

#ifdef FF_BOUNDARY_PTR_LEFT
    for (; i_pixel < KERNEL_LEN; ++i_pixel) {
        for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
            float sum = 0.0;

            sum = kernel->coefs[0] * inptr[pixel_stride * i_pixel + outer_stride * i_outer];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                float left;
                if ((int)i_pixel - (int)k < 0)
                    left = in_border_left[i_outer * outer_stride +
                                          (KERNEL_LEN - (int)k + (int)i_pixel) * borderptr_outer_stride];
                else
                    left = inptr[(i_pixel - k) * pixel_stride + outer_stride * i_outer];

#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (inptr[(i_pixel + k) * pixel_stride + outer_stride * i_outer] + left);
#else
                sum += kernel->coefs[k] * (inptr[(i_pixel + k) * pixel_stride + outer_stride * i_outer] - left);
#endif
            }

            tmp[n_outer * i_pixel + i_outer] = sum;
        }
    }
#endif

// 'valid'
#if defined(FF_BOUNDARY_MIRROR_RIGHT) || defined(FF_BOUNDARY_PTR_RIGHT)
    const unsigned int end = n_pixels - KERNEL_LEN;
#else
    const unsigned int end = n_pixels;
#endif
    for (; i_pixel < end; ++i_pixel) {
        const unsigned tmpidx = i_pixel % (KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer;

        for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
            float sum = 0.0;

            sum = kernel->coefs[0] * inptr[pixel_stride * i_pixel + outer_stride * i_outer];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                int offset_left = (i_pixel - k) * pixel_stride + outer_stride * i_outer;
#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] *
                       (inptr[(i_pixel + k) * pixel_stride + outer_stride * i_outer] + inptr[offset_left]);
#else
                sum += kernel->coefs[k] *
                       (inptr[(i_pixel + k) * pixel_stride + outer_stride * i_outer] - inptr[offset_left]);
#endif
            }

            tmpptr[i_outer] = sum;
        }

#ifdef FF_BOUNDARY_OPTIMISTIC_LEFT
        if (i_pixel < KERNEL_LEN)
            continue;
#endif

        const unsigned writeidx = (i_pixel + 1) % (KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer;
        // FIXME: outer_stride
        memcpy(outptr + (i_pixel - KERNEL_LEN) * pixel_stride, writeptr, n_outer * sizeof(float));
    }

// right border
#ifdef FF_BOUNDARY_MIRROR_RIGHT
    for (; i_pixel < n_pixels; ++i_pixel) {
        const unsigned tmpidx = i_pixel % (KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer;

        for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
            float sum = 0.0;

            sum = kernel->coefs[0] * inptr[pixel_stride * i_pixel + outer_stride * i_outer];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                unsigned int offset_left, offset_right;
                if (-(int)k + (int)i_pixel < 0)
                    offset_left = -i_pixel + k;
                else
                    offset_left = i_pixel - k;

                if (k + i_pixel >= n_pixels)
                    offset_right = n_pixels - ((k + i_pixel) % n_pixels) - 2;
                else
                    offset_right = i_pixel + k;

#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (inptr[offset_right * pixel_stride + outer_stride * i_outer] +
                                           inptr[offset_left * pixel_stride + outer_stride * i_outer]);
#else
                sum += kernel->coefs[k] * (inptr[offset_right * pixel_stride + outer_stride * i_outer] -
                                           inptr[offset_left * pixel_stride + outer_stride * i_outer]);
#endif
            }

            tmpptr[i_outer] = sum;
        }

        const unsigned writeidx = (i_pixel + 1) % (KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer;
        // FIXME: outer_stride
        memcpy(outptr + (i_pixel - KERNEL_LEN) * pixel_stride, writeptr, n_outer * sizeof(float));
    }
#endif

#ifdef FF_BOUNDARY_PTR_RIGHT
    for (; i_pixel < n_pixels; ++i_pixel) {
        const unsigned tmpidx = i_pixel % (KERNEL_LEN + 1);
        float *tmpptr = tmp + tmpidx * n_outer;

        for (unsigned int i_outer = 0; i_outer < n_outer; ++i_outer) {
            float sum = 0.0;

            sum = kernel->coefs[0] * inptr[pixel_stride * i_pixel + outer_stride * i_outer];

            for (unsigned int k = 1; k <= KERNEL_LEN; ++k) {
                float right;
                if (k + i_pixel >= n_pixels)
                    right =
                        in_border_right[i_outer * outer_stride + ((k + i_pixel) % n_pixels) * borderptr_outer_stride];
                else
                    right = inptr[(i_pixel + k) * pixel_stride + outer_stride * i_outer];

#ifdef FF_KERNEL_SYMMETRIC
                sum += kernel->coefs[k] * (right + inptr[(i_pixel - k) * pixel_stride + outer_stride * i_outer]);
#else
                sum += kernel->coefs[k] * (right - inptr[(i_pixel - k) * pixel_stride + outer_stride * i_outer]);
#endif
            }

            tmpptr[i_outer] = sum;
        }

        const unsigned writeidx = (i_pixel + 1) % (KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer;
        // FIXME: outer_stride
        memcpy(outptr + (i_pixel - KERNEL_LEN) * pixel_stride, writeptr, n_outer * sizeof(float));
    }
#endif

    for (unsigned i = 0; i < KERNEL_LEN; ++i) {
        unsigned pixel = n_pixels + i;
        const unsigned writeidx = (pixel + 1) % (KERNEL_LEN + 1);
        float *writeptr = tmp + writeidx * n_outer;
        // FIXME: outer_stride
        memcpy(outptr + (pixel - KERNEL_LEN) * pixel_stride, writeptr, n_outer * sizeof(float));
    }

    fastfilters_memory_free(tmp);
    return true;
}

#undef symmetry_name
#undef boundary_name
#undef boundary_name_left
#undef boundary_name_right
#undef KERNEL_LEN
#undef FNAME

#endif
