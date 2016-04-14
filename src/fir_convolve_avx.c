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

BOOST_PP_REPEAT(FF_UNROLL, fname_extern_define_all, ~)

// HACK: fix syntax hilighting :-)
#ifndef dummy
#define dummy
#undef dummy
#endif

struct impl_fn_jmptbl_selection {
    impl_fn_t *jmptbl;
    fastfilters_border_treatment_t left_border;
    fastfilters_border_treatment_t right_border;
    bool is_symmetric;
};

#define FF_KERNEL_LEN_RUNTIME
#include "fir_convolve_avx_impl.c"
#undef FF_KERNEL_LEN_RUNTIME

#define KLEN_SUFFIX(n) BOOST_PP_IF(BOOST_PP_EQUAL(BOOST_PP_INC(n), FF_UNROLL), N, BOOST_PP_INC(n))

#define DECL_JMPTBL_MACRO(z, n, text)                                                                                  \
    fname(BOOST_PP_TUPLE_ELEM(5, 0, text), BOOST_PP_TUPLE_ELEM(5, 1, text), BOOST_PP_TUPLE_ELEM(5, 2, text),           \
          BOOST_PP_TUPLE_ELEM(5, 3, text), BOOST_PP_TUPLE_ELEM(5, 4, text), KLEN_SUFFIX(n))

#define DECL_JMPTBL(outer, left_border, right_border, symmetric, fma)                                                  \
    static impl_fn_t BOOST_PP_CAT(g_, fname(outer, left_border, right_border, symmetric, fma, _tbl))[] = {             \
        BOOST_PP_ENUM(FF_UNROLL, DECL_JMPTBL_MACRO, (outer, left_border, right_border, symmetric, fma))};

#define DECL_JMPTBL_CALL(r, prod)                                                                                      \
    DECL_JMPTBL(BOOST_PP_TUPLE_ELEM(5, 0, prod), BOOST_PP_TUPLE_ELEM(5, 1, prod), BOOST_PP_TUPLE_ELEM(5, 2, prod),     \
                BOOST_PP_TUPLE_ELEM(5, 3, prod), BOOST_PP_TUPLE_ELEM(5, 4, prod))

BOOST_PP_LIST_FOR_EACH_PRODUCT(DECL_JMPTBL_CALL, 5,
                               (l_outer, l_border, l_border, l_symmetric, (param_avxfma, BOOST_PP_NIL)));

#define DEFINE_JMPTBL_STRUCT(outer, x, y, symmetric)                                                                   \
    {                                                                                                                  \
        .jmptbl = BOOST_PP_CAT(g_, fname(outer, x, y, symmetric, param_avxfma, _tbl)), .left_border = ENUM_BORDER(x),  \
        .right_border = ENUM_BORDER(y), .is_symmetric = BOOST_PP_IF(symmetric, true, false)                            \
    }                                                                                                                  \
    ,

#define DEFINE_JMPTBL_STRUCT_M(r, prod)                                                                                \
    DEFINE_JMPTBL_STRUCT(BOOST_PP_TUPLE_ELEM(4, 0, prod), BOOST_PP_TUPLE_ELEM(4, 1, prod),                             \
                         BOOST_PP_TUPLE_ELEM(4, 2, prod), BOOST_PP_TUPLE_ELEM(4, 3, prod))

// HACK: fix syntax hilighting :-)
#ifndef dummy
#define dummy
#undef dummy
#endif

static const struct impl_fn_jmptbl_selection jmptbls_inner[] = {
    BOOST_PP_LIST_FOR_EACH_PRODUCT(DEFINE_JMPTBL_STRUCT_M, 4, ((0, BOOST_PP_NIL), l_border, l_border, l_symmetric))};

static const struct impl_fn_jmptbl_selection jmptbls_outer[] = {
    BOOST_PP_LIST_FOR_EACH_PRODUCT(DEFINE_JMPTBL_STRUCT_M, 4, ((1, BOOST_PP_NIL), l_border, l_border, l_symmetric))};

static impl_fn_t find_fn(fastfilters_kernel_fir_t kernel, fastfilters_border_treatment_t left_border,
                         fastfilters_border_treatment_t right_border, const struct impl_fn_jmptbl_selection *jmptbls,
                         size_t n_jmptbls)
{
    if (kernel->len == 0)
        return NULL;

    impl_fn_t *jmptbl = NULL;

    for (unsigned int i = 0; i < n_jmptbls; ++i) {
        if (left_border != jmptbls[i].left_border)
            continue;
        if (right_border != jmptbls[i].right_border)
            continue;
        if (kernel->is_symmetric != jmptbls[i].is_symmetric)
            continue;
        jmptbl = jmptbls[i].jmptbl;
        break;
    }

    if (jmptbl == NULL)
        return NULL;

    if (kernel->len > FF_UNROLL)
        return jmptbl[FF_UNROLL];
    else
        return jmptbl[kernel->len - 1];
}

#define APPEND_AVXFMA(x) BOOST_PP_CAT3(x, _, fname_avxfma(param_avxfma))

bool APPEND_AVXFMA(fastfilters_fir_convolve_fir_inner)(const float *inptr, size_t n_pixels, size_t pixel_stride,
                                                       size_t n_outer, size_t outer_stride, float *outptr,
                                                       fastfilters_kernel_fir_t kernel,
                                                       fastfilters_border_treatment_t left_border,
                                                       fastfilters_border_treatment_t right_border,
                                                       const float *borderptr_left, const float *borderptr_right,
                                                       size_t border_outer_stride)
{
    impl_fn_t fn = find_fn(kernel, left_border, right_border, jmptbls_inner, ARRAY_LENGTH(jmptbls_inner));

    if (fn == NULL)
        return false;

    return fn(inptr, borderptr_left, borderptr_right, n_pixels, pixel_stride, n_outer, outer_stride, outptr,
              outer_stride, border_outer_stride, kernel);
}

bool APPEND_AVXFMA(fastfilters_fir_convolve_fir_outer)(const float *inptr, size_t n_pixels, size_t pixel_stride,
                                                       size_t n_outer, size_t outer_stride, float *outptr,
                                                       fastfilters_kernel_fir_t kernel,
                                                       fastfilters_border_treatment_t left_border,
                                                       fastfilters_border_treatment_t right_border,
                                                       const float *borderptr_left, const float *borderptr_right,
                                                       size_t border_outer_stride)
{
    impl_fn_t fn = find_fn(kernel, left_border, right_border, jmptbls_outer, ARRAY_LENGTH(jmptbls_outer));

    if (fn == NULL)
        return false;

    return fn(inptr, borderptr_left, borderptr_right, n_pixels, pixel_stride, n_outer, outer_stride, outptr,
              outer_stride, border_outer_stride, kernel);
}
