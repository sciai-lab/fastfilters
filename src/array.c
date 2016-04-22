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

DLL_PUBLIC fastfilters_array2d_t *fastfilters_array2d_alloc(size_t n_x, size_t n_y, size_t channels)
{
    fastfilters_array2d_t *result = NULL;

    result = fastfilters_memory_alloc(sizeof(*result));
    if (!result)
        goto error_out;

    result->n_x = n_x;
    result->n_y = n_y;
    result->stride_x = channels;
    result->stride_y = channels * n_x;
    result->n_channels = channels;
    result->ptr = fastfilters_memory_alloc(channels * n_y * n_x * sizeof(float));
    if (!result->ptr)
        goto error_out;

    return result;

error_out:
    if (result) {
        if (result->ptr)
            fastfilters_memory_free(result->ptr);
        fastfilters_memory_free(result);
    }
    return NULL;
}

DLL_PUBLIC void fastfilters_array2d_free(fastfilters_array2d_t *v)
{
    fastfilters_memory_free(v->ptr);
    fastfilters_memory_free(v);
}