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

#ifndef CPUID_HXX_
#define CPUID_HXX_ 1

#include "config.h"
#include <array>

#ifdef HAVE_CPUID_H
#include <cpuid.h>
#elif defined(HAVE_ASM_CPUID)
#include "clang_cpuid.h"
#endif

#ifdef HAVE_CPUIDEX
#include <intrin.h>
#endif

typedef std::array<unsigned int, 4> cpuid_t;

static inline int get_cpuid(unsigned int level, cpuid_t &id)
{
#if defined(HAVE_CPUID_H) || defined(HAVE_ASM_CPUID)

    if ((unsigned int)__get_cpuid_max(0, NULL) < level)
        return 0;

    __cpuid_count(level, 0, id[0], id[1], id[2], id[3]);
    return 1;

#elif defined(HAVE_CPUIDEX)
    int cpuid[4];
    __cpuidex(cpuid, level, 0);

    for (unsigned int i = 0; i < 4; ++i)
        id[i] = (unsigned int)cpuid[i];

    return 1;

#else
#error "No known way to query for CPUID."
#endif
}

#endif