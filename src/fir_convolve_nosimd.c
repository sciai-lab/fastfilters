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

// hack to make waf dependency tracking + warning generation work correctly
// this creates fours functions called fir_convolve_impl_{mirror,optimistic}_{anti,}symmetric0 that is never used
#define BOOST_PP_ITERATION() 0
#include "fir_convolve_nosimd_impl.h"
#undef BOOST_PP_ITERATION

#define FF_UNROLL 20

// yo dawg, i herd you like #includes so i put an #include in your #include so you can #include while you #include
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/list/cat.hpp>

#define BOOST_PP_ITERATION_LIMITS (1, FF_UNROLL)
#define BOOST_PP_FILENAME_1 "fir_convolve_nosimd_impl.h"
#include BOOST_PP_ITERATE()
#undef BOOST_PP_ITERATION_LIMITS
#undef BOOST_PP_FILENAME_1

typedef bool (*impl_fn_t)(const float *, size_t, size_t, size_t, size_t, float *, const float *);

#define IMPL_FNAME(z, n, text) &BOOST_PP_CAT(BOOST_PP_CAT(fir_convolve_impl_, text), BOOST_PP_INC(n))

static impl_fn_t impl_fn_symmetric_mirror[] = {BOOST_PP_ENUM(FF_UNROLL, IMPL_FNAME, mirror_symmetric)};

static impl_fn_t impl_fn_antisymmetric_mirror[] = {BOOST_PP_ENUM(FF_UNROLL, IMPL_FNAME, mirror_antisymmetric)};

static impl_fn_t impl_fn_symmetric_optimistic[] = {BOOST_PP_ENUM(FF_UNROLL, IMPL_FNAME, optimistic_symmetric)};

static impl_fn_t impl_fn_antisymmetric_optimistic[] = {BOOST_PP_ENUM(FF_UNROLL, IMPL_FNAME, optimistic_antisymmetric)};

bool fastfilters_fir_convolve_fir_inner(const float *inptr, size_t n_pixels, size_t pixel_stride, size_t n_outer,
                                        size_t outer_stride, float *outptr, fastfilters_kernel_fir_t kernel,
                                        fastfilters_border_treatment_t border)
{
    // disable warnings about the function that is never used and just there to create
    // compiler warnings
    (void)fir_convolve_impl_optimistic_symmetric0(NULL, 0, 0, 0, 0, NULL, NULL);
    (void)fir_convolve_impl_optimistic_antisymmetric0(NULL, 0, 0, 0, 0, NULL, NULL);
    (void)fir_convolve_impl_mirror_symmetric0(NULL, 0, 0, 0, 0, NULL, NULL);
    (void)fir_convolve_impl_mirror_antisymmetric0(NULL, 0, 0, 0, 0, NULL, NULL);

    if (kernel->len == 0)
        return false;

    if (kernel->len > FF_UNROLL)
        return false;

    impl_fn_t *jmptbl = NULL;

    switch (border) {
    case FASTFILTERS_BORDER_MIRROR:
        if (kernel->is_symmetric)
            jmptbl = impl_fn_symmetric_mirror;
        else
            jmptbl = impl_fn_antisymmetric_mirror;
        break;
    case FASTFILTERS_BORDER_OPTIMISTIC:
        if (kernel->is_symmetric)
            jmptbl = impl_fn_symmetric_optimistic;
        else
            jmptbl = impl_fn_antisymmetric_optimistic;
        break;
    default:
        return false;
    }

    return jmptbl[kernel->len - 1](inptr, n_pixels, pixel_stride, n_outer, outer_stride, outptr, kernel->coefs);
}
