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
#include "config.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "fir_convolve_avx_common.h"

#if !defined(FF_KERNEL_LEN) && !defined(FF_KERNEL_LEN_RUNTIME)
#error !defined(FF_KERNEL_LEN) && !defined(FF_KERNEL_LEN_RUNTIME)
#endif

#if !defined(FF_BOUNDARY_OPTIMISTIC_LEFT) && !defined(FF_BOUNDARY_MIRROR_LEFT) && !defined(FF_BOUNDARY_PTR_LEFT)

#define FF_BOUNDARY_OPTIMISTIC_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_OPTIMISTIC_LEFT

#define FF_BOUNDARY_MIRROR_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_MIRROR_LEFT

#define FF_BOUNDARY_PTR_LEFT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_PTR_LEFT

#elif !defined(FF_BOUNDARY_OPTIMISTIC_RIGHT) && !defined(FF_BOUNDARY_MIRROR_RIGHT) && !defined(FF_BOUNDARY_PTR_RIGHT)

#define FF_BOUNDARY_OPTIMISTIC_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_OPTIMISTIC_RIGHT

#define FF_BOUNDARY_MIRROR_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_MIRROR_RIGHT

#define FF_BOUNDARY_PTR_RIGHT
#include "fir_convolve_avx_impl.c"
#undef FF_BOUNDARY_PTR_RIGHT

#elif !defined(FF_KERNEL_SYMMETRIC) && !defined(FF_KERNEL_ANTISYMMETRIC)

#define FF_KERNEL_SYMMETRIC
#include "fir_convolve_avx_impl.c"
#undef FF_KERNEL_SYMMETRIC

#define FF_KERNEL_ANTISYMMETRIC
#include "fir_convolve_avx_impl.c"
#undef FF_KERNEL_ANTISYMMETRIC

#else

#ifdef FF_BOUNDARY_MIRROR_LEFT
#define param_boundary_left 0
#elif defined(FF_BOUNDARY_OPTIMISTIC_LEFT)
#define param_boundary_left 1
#elif defined(FF_BOUNDARY_PTR_LEFT)
#define param_boundary_left 2
#else
#error "Unknown border left"
#endif

#ifdef FF_BOUNDARY_MIRROR_RIGHT
#define param_boundary_right 0
#elif defined(FF_BOUNDARY_OPTIMISTIC_RIGHT)
#define param_boundary_right 1
#elif defined(FF_BOUNDARY_PTR_RIGHT)
#define param_boundary_right 2
#else
#error "Unknown border right"
#endif

#ifdef FF_KERNEL_SYMMETRIC
#define param_symm 1
#elif defined(FF_KERNEL_ANTISYMMETRIC)
#define param_symm 0
#else
#error "Unknown symmetry"
#endif

#ifdef FF_KERNEL_LEN_RUNTIME
#define FF_KERNEL_LEN kernel->len
#define FF_KERNEL_LEN_FNAME N
#else
#define FF_KERNEL_LEN_FNAME FF_KERNEL_LEN
#endif

bool DLL_LOCAL fname(0, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
    (void)inptr;
    (void)in_border_left;
    (void)in_border_right;
    (void)n_pixels;
    (void)pixel_stride;
    (void)n_outer;
    (void)outer_stride;
    (void)outptr;
    (void)outptr_outer_stride;
    (void)borderptr_outer_stride;
    (void)kernel;
    return false;
}

bool DLL_LOCAL fname(1, param_boundary_left, param_boundary_right, param_symm, param_avxfma,
                     FF_KERNEL_LEN_FNAME)(const float *inptr, const float *in_border_left, const float *in_border_right,
                                          size_t n_pixels, size_t pixel_stride, size_t n_outer, size_t outer_stride,
                                          float *outptr, size_t outptr_outer_stride, size_t borderptr_outer_stride,
                                          const fastfilters_kernel_fir_t kernel)
{
    (void)inptr;
    (void)in_border_left;
    (void)in_border_right;
    (void)n_pixels;
    (void)pixel_stride;
    (void)n_outer;
    (void)outer_stride;
    (void)outptr;
    (void)outptr_outer_stride;
    (void)borderptr_outer_stride;
    (void)kernel;
    return false;
}

#undef param_symm
#undef param_boundary_left
#undef param_boundary_right
#undef FF_KERNEL_LEN_FNAME

#endif
