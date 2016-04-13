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

#ifndef FASTFILTERS_COMMON_H
#define FASTFILTERS_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "fastfilters.h"
#include "config.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef HAVE_BUILTIN_EXPECT
#define likely(x) __builtin_expect(x, true)
#define unlikely(x) __builtin_expect(x, false)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

struct _fastfilters_kernel_fir_t {
    size_t len;
    bool is_symmetric;
    float *coefs;
};

typedef enum {
    FASTFILTERS_BORDER_MIRROR,
    FASTFILTERS_BORDER_OPTIMISTIC,
    FASTFILTERS_BORDER_PTR
} fastfilters_border_treatment_t;

void DLL_LOCAL fastfilters_cpu_init(void);
void DLL_LOCAL fastfilters_linalg_init(void);

void DLL_LOCAL fastfilters_memory_init(fastfilters_alloc_fn_t alloc_fn, fastfilters_free_fn_t free_fn);

void DLL_LOCAL *fastfilters_memory_alloc(size_t size);
void DLL_LOCAL fastfilters_memory_free(void *ptr);

void DLL_LOCAL *fastfilters_memory_align(size_t alignment, size_t size);
void DLL_LOCAL fastfilters_memory_align_free(void *ptr);

void DLL_LOCAL fastfilters_fir_init(void);

bool DLL_LOCAL fastfilters_fir_convolve_fir_inner(const float *inptr, size_t n_pixels, size_t pixel_stride,
                                                  size_t n_outer, size_t outer_stride, float *outptr,
                                                  fastfilters_kernel_fir_t kernel,
                                                  fastfilters_border_treatment_t left_border,
                                                  fastfilters_border_treatment_t right_border,
                                                  const float *borderptr_left, const float *borderptr_right,
                                                  size_t border_outer_stride);
bool DLL_LOCAL fastfilters_fir_convolve_fir_outer(const float *inptr, size_t n_pixels, size_t pixel_stride,
                                                  size_t n_outer, size_t outer_stride, float *outptr,
                                                  fastfilters_kernel_fir_t kernel,
                                                  fastfilters_border_treatment_t left_border,
                                                  fastfilters_border_treatment_t right_border,
                                                  const float *borderptr_left, const float *borderptr_right,
                                                  size_t border_outer_stride);

#ifdef __cplusplus
}
#endif

#endif
