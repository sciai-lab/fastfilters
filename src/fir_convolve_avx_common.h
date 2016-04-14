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

#ifndef FIR_CONVOLVE_AVX_COMMON_H
#define FIR_CONVOLVE_AVX_COMMON_H

#if defined(__AVX__) && defined(__FMA__)
#define param_avxfma 1
#elif defined(__AVX__)
#define param_avxfma 0
#else
#error "fir_convolve_avx*.c need to be compiled with AVX support."
#endif

#define FF_UNROLL 20

#include <boost/preprocessor/library.hpp>
#define N_BORDER_TYPES 3

#define BOOST_PP_CAT3(a, b, c) BOOST_PP_CAT(a, BOOST_PP_CAT(b, c))
#define BOOST_PP_CAT5(a, b, c, d, e) BOOST_PP_CAT3(BOOST_PP_CAT3(a, b, c), d, e)
#define BOOST_PP_CAT9(a, b, c, d, e, f, g, h, i) BOOST_PP_CAT5(BOOST_PP_CAT5(a, b, c, d, e), f, g, h, i)

#define fname_outer(outer) BOOST_PP_IF(outer, fir_convolve_outer_impl, fir_convolve_impl)
#define fname_border(x) BOOST_PP_CAT(border_, x)
#define fname_symmetric(x) BOOST_PP_IF(x, symmetric, antisymmetric)
#define fname_aligned(x) BOOST_PP_IF(x, aligned, unaligned)
#define fname_avxfma(x) BOOST_PP_IF(x, avxfma, avx)

#define fname(outer, left_border, right_border, symmetric, fma, n)                                                     \
    BOOST_PP_CAT(BOOST_PP_CAT9(fname_outer(outer), _, fname_border(left_border), _, fname_border(right_border), _,     \
                               fname_symmetric(symmetric), _, fname_avxfma(fma)),                                      \
                 n)

#define fname_extern(outer, left_border, right_border, symmetric, fma, n)                                              \
    extern bool DLL_LOCAL fname(outer, left_border, right_border, symmetric, fma,                                      \
                                n)(const float *, const float *, const float *, size_t, size_t, size_t, size_t,        \
                                   float *, size_t, size_t, const fastfilters_kernel_fir_t);

#define l_outer (0, (1, BOOST_PP_NIL))
#define l_border (0, (1, (2, BOOST_PP_NIL)))
#define l_symmetric (0, (1, BOOST_PP_NIL))

#define fname_extern_define_macro(r, prod)                                                                             \
    fname_extern(BOOST_PP_TUPLE_ELEM(5, 0, prod), BOOST_PP_TUPLE_ELEM(5, 1, prod), BOOST_PP_TUPLE_ELEM(5, 2, prod),    \
                 BOOST_PP_TUPLE_ELEM(5, 3, prod), param_avxfma, BOOST_PP_TUPLE_ELEM(5, 4, prod))
#define fname_extern_define_all(z, n, text)                                                                            \
    BOOST_PP_LIST_FOR_EACH_PRODUCT(fname_extern_define_macro, 5,                                                       \
                                   (l_outer, l_border, l_border, l_symmetric, (n, BOOST_PP_NIL)))

#define ENUM_BORDER(x) BOOST_PP_CAT(border_enum_, x)

#endif