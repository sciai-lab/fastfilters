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

#include <stdlib.h>
#include <assert.h>

#define ALIGN_MAGIC 0xd2ac461d9c25ee00

static fastfilters_alloc_fn_t g_alloc_fn = NULL;
static fastfilters_free_fn_t g_free_fn = NULL;

void fastfilters_memory_init(fastfilters_alloc_fn_t alloc_fn, fastfilters_free_fn_t free_fn)
{
    if (alloc_fn)
        g_alloc_fn = alloc_fn;
    else
        g_alloc_fn = malloc;

    if (free_fn)
        g_free_fn = free_fn;
    else
        g_free_fn = free;
}

void *fastfilters_memory_alloc(size_t size)
{
    return g_alloc_fn(size);
}

void fastfilters_memory_free(void *ptr)
{
    g_free_fn(ptr);
}

void *fastfilters_memory_align(size_t alignment, size_t size)
{
    assert(alignment < 0xff);
    void *ptr = fastfilters_memory_alloc(size + alignment + 8);

    if (!ptr)
        return NULL;

    uintptr_t ptr_i = (uintptr_t)ptr;
    ptr_i += 8 + alignment - 1;
    ptr_i &= ~(alignment - 1);

    uintptr_t ptr_diff = ptr_i - (uintptr_t)ptr;

    void *ptr_aligned = (void *)ptr_i;
    uint64_t *ptr_magic = (uint64_t *)(ptr_i - 8);

    *ptr_magic = ALIGN_MAGIC | (ptr_diff & 0xff);

    return ptr_aligned;
}

void fastfilters_memory_align_free(void *ptr)
{
    char *ptr_cast = (char *)ptr;
    uint64_t magic = *(uint64_t *)(ptr_cast - 8);

    assert((magic & ~0xff) == ALIGN_MAGIC);

    ptr_cast -= magic & 0xff;
    fastfilters_memory_free(ptr_cast);
}