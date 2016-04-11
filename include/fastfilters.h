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

#ifndef FASTFILTERS_H
#define FASTFILTERS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef struct _fastfilters_kernel_fir_t *fastfilters_kernel_fir_t;

typedef enum { FASTFILTERS_CPU_AVX, FASTFILTERS_CPU_FMA, FASTFILTERS_CPU_AVX2 } fastfilters_cpu_feature_t;

typedef enum { FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_OPTIMISTIC } fastfilters_border_treatment_t;

typedef void *(*fastfilters_alloc_fn_t)(size_t size);
typedef void (*fastfilters_free_fn_t)(void *);

void fastfilters_init(fastfilters_alloc_fn_t alloc_fn, fastfilters_free_fn_t free_fn);

bool fastfilters_cpu_check(fastfilters_cpu_feature_t feature);
bool fastfilters_cpu_enable(fastfilters_cpu_feature_t feature, bool enable);

fastfilters_kernel_fir_t fastfilters_kernel_fir_gaussian(unsigned int order, double sigma);
void fastfilters_kernel_fir_free(fastfilters_kernel_fir_t kernel);

bool fastfilters_fir_convolve2d(const float *inptr, float *outptr, size_t n_x, size_t n_y, size_t stride_x,
                                size_t stride_y, size_t x0, size_t y0, size_t x1, size_t y1);

void fastfilters_linalg_ev2d(const float *xx, const float *xy, const float *yy, float *ev_small, float *ev_big,
                             const size_t len);

#ifdef __cplusplus
}
#endif

#endif