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

#ifndef XGETBV_HXX_
#define XGETBV_HXX_ 1

#define cpuid_bit_XSAVE 0x04000000
#define cpuid_bit_OSXSAVE 0x08000000

typedef unsigned long long xgetbv_t;

#if defined(HAVE_ASM_XGETBV)

#include "cpuid.hxx"

static inline xgetbv_t xgetbv()
{
    unsigned int index = 0;
    unsigned int eax, edx;

    // CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1
    cpuid_t cpuid;
    int res = get_cpuid(1, cpuid);

    if (!res)
        return 0;

    if ((cpuid[2] & cpuid_bit_XSAVE) != cpuid_bit_XSAVE)
        return 0;
    if ((cpuid[2] & cpuid_bit_OSXSAVE) != cpuid_bit_OSXSAVE)
        return 0;

    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));

    return ((unsigned long long)edx << 32) | eax;
}

#elif defined(HAVE_INTRIN_XGETBV)

static inline xgetbv_t xgetbv()
{
    return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
}

#else
#error "No known way to use xgetbv."
#endif

#endif
