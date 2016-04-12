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

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "fastfilters.h"
#include "common.h"

typedef bool (*fir_convolve_fn_t)(const float *, size_t, size_t, size_t, size_t, float *, fastfilters_kernel_fir_t,
                                  fastfilters_border_treatment_t, fastfilters_border_treatment_t);

static fir_convolve_fn_t g_convolve_inner = NULL;
static fir_convolve_fn_t g_convolve_outer = NULL;

void fastfilters_fir_init(void)
{
    g_convolve_outer = &fastfilters_fir_convolve_fir_outer;
    g_convolve_inner = &fastfilters_fir_convolve_fir_inner;
}

bool fastfilters_fir_convolve2d(const fastfilters_array2d_t *inarray, const fastfilters_kernel_fir_t kernelx,
                                const fastfilters_kernel_fir_t kernely, float *outptr, size_t x0, size_t y0, size_t x1,
                                size_t y1)
{
    if (x0 == 0 && y0 == 0 && x1 == 0 && y1 == 0) {
        if (!g_convolve_inner(inarray->ptr, inarray->n_x, inarray->stride_x, inarray->n_y, inarray->stride_y, outptr,
                              kernelx, FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_MIRROR))
            return false;

        return g_convolve_outer(outptr, inarray->n_y, inarray->stride_y, inarray->n_x, inarray->stride_x, outptr,
                                kernely, FASTFILTERS_BORDER_MIRROR, FASTFILTERS_BORDER_MIRROR);
    }

    return false;
}