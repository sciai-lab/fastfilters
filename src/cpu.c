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
#include "config.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef HAVE_CPUID_H
#include <cpuid.h>
#elif defined(HAVE_ASM_CPUID)
#include "clang_cpuid.h"
#endif

#ifdef HAVE_CPUIDEX
#include <intrin.h>
#endif

#define cpuid_bit_XSAVE 0x04000000
#define cpuid_bit_OSXSAVE 0x08000000
#define cpuid_bit_AVX 0x10000000
#define cpuid_bit_FMA 0x00001000
#define cpuid7_bit_AVX2 0x00000020

#define xcr0_bit_XMM 0x00000002
#define xcr0_bit_YMM 0x00000004

typedef struct {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
} cpuid_t;

typedef unsigned long long xgetbv_t;

static inline int get_cpuid(unsigned int level, cpuid_t *id)
{
#if defined(HAVE_CPUID_H) || defined(HAVE_ASM_CPUID)

    if ((unsigned int)__get_cpuid_max(0, NULL) < level)
        return 0;

    __cpuid_count(level, 0, id->eax, id->ebx, id->ecx, id->edx);
    return 1;

#elif defined(HAVE_CPUIDEX)
    int cpuid[4];
    __cpuidex(cpuid, level, 0);

    id->eax = id[0];
    id->ebx = id[1];
    id->ecx = id[2];
    id->edx = id[3];

    return 1;

#else
#error "No known way to query for CPUID."
#endif
}

#if defined(HAVE_ASM_XGETBV)

static inline xgetbv_t xgetbv()
{
    unsigned int index = 0;
    unsigned int eax, edx;

    // CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1
    cpuid_t cpuid;
    int res = get_cpuid(1, &cpuid);

    if (!res)
        return 0;

    if ((cpuid.ecx & cpuid_bit_XSAVE) != cpuid_bit_XSAVE)
        return 0;
    if ((cpuid.ecx & cpuid_bit_OSXSAVE) != cpuid_bit_OSXSAVE)
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

#if defined(HAVE_GNU_CPU_SUPPORTS_AVX2)

static bool _supports_avx2()
{
    if (__builtin_cpu_supports("avx2"))
        return true;
    else
        return false;
}

#else

static bool _supports_avx2()
{
    cpuid_t cpuid;

    // CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1
    int res = get_cpuid(7, &cpuid);

    if (!res)
        return false;

    if ((cpuid.ebx & cpuid7_bit_AVX2) != cpuid7_bit_AVX2)
        return false;

    xgetbv_t xcr0;
    xcr0 = xgetbv();

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    if ((xcr0 & xcr0_bit_XMM) != xcr0_bit_XMM)
        return false;
    if ((xcr0 & xcr0_bit_YMM) != xcr0_bit_YMM)
        return false;

    return true;
}

#endif

#if defined(HAVE_GNU_CPU_SUPPORTS_AVX)

static bool _supports_avx()
{
    if (__builtin_cpu_supports("avx"))
        return true;
    else
        return false;
}

#else

static bool _supports_avx()
{
    cpuid_t cpuid;

    // CPUID.(EAX=01H, ECX=0H):ECX.AVX[bit 28]==1
    int res = get_cpuid(1, &cpuid);

    if (!res)
        return false;

    if ((cpuid.ecx & cpuid_bit_AVX) != cpuid_bit_AVX)
        return false;

    xgetbv_t xcr0;
    xcr0 = xgetbv();

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    if (((xcr0 & 6) != 6))
        return false;
    return true;
}

#endif

#if defined(HAVE_GNU_CPU_SUPPORTS_FMA)

static bool _supports_fma()
{
    if (__builtin_cpu_supports("fma") && __builtin_cpu_supports("avx"))
        return true;
    else
        return false;
}

#else

static bool _supports_fma()
{
    cpuid_t cpuid;

    if (!_supports_avx())
        return false;

    // check for CPU FMA support: CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1
    int res = get_cpuid(1, &cpuid);

    if (!res)
        return false;

    if ((cpuid.ecx & cpuid_bit_FMA) != cpuid_bit_FMA)
        return false;

    return true;
}

#endif

static bool g_supports_avx = false;
static bool g_supports_fma = false;
static bool g_supports_avx2 = false;

void fastfilters_cpu_init(void)
{
    g_supports_avx = _supports_avx();
    g_supports_fma = _supports_fma();
    g_supports_avx2 = _supports_avx2();
}

bool fastfilters_cpu_enable(fastfilters_cpu_feature_t feature, bool enable)
{
    switch (feature) {
    case FASTFILTERS_CPU_AVX:
        if (enable)
            g_supports_avx = _supports_avx();
        else
            g_supports_avx = false;
        break;
    case FASTFILTERS_CPU_FMA:
        if (enable)
            g_supports_fma = _supports_fma();
        else
            g_supports_fma = false;
        break;
    case FASTFILTERS_CPU_AVX2:
        if (enable)
            g_supports_avx2 = _supports_avx2();
        else
            g_supports_avx2 = false;
        break;
    default:
        return false;
    }

    return fastfilters_cpu_check(feature);
}

bool fastfilters_cpu_check(fastfilters_cpu_feature_t feature)
{
    switch (feature) {
    case FASTFILTERS_CPU_AVX:
        return g_supports_avx;
    case FASTFILTERS_CPU_FMA:
        return g_supports_fma;
    case FASTFILTERS_CPU_AVX2:
        return g_supports_avx2;
    default:
        return false;
    }
}