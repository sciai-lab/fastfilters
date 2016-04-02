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
#include "fastfilters.hxx"

#include "config.h"
#include "cpuid.hxx"
#include "xgetbv.hxx"

namespace fastfilters
{

namespace detail
{

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
    int res = get_cpuid(7, cpuid);

    if (!res)
        return false;

    if ((cpuid[1] & (1 << 5)) != (1 << 5))
        return false;

    xgetbv_t xcr0;
    xcr0 = xgetbv();

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    if (((xcr0 & 6) != 6))
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
    int res = get_cpuid(1, cpuid);

    if (!res)
        return false;

    if ((cpuid[2] & (1 << 28)) != (1 << 28))
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
    int res = get_cpuid(1, cpuid);

    if (!res)
        return false;

    if ((cpuid[2] & (1 << 12)) != (1 << 12))
        return false;

    return true;
}

#endif

static bool supports_avx = _supports_avx();
static bool supports_fma = _supports_fma();
static bool supports_avx2 = _supports_avx2();

bool cpu_has_avx2()
{
    return supports_avx2;
}

bool cpu_has_avx()
{
    return supports_avx;
}

bool cpu_has_avx_fma()
{
    return supports_fma;
}

bool cpu_enable_avx2(bool enable)
{
    if (enable)
        supports_avx2 = _supports_avx2();
    else
        supports_avx2 = false;

    return supports_avx2;
}

bool cpu_enable_avx(bool enable)
{
    if (enable)
        supports_avx = _supports_avx();
    else
        supports_avx = false;

    return supports_avx;
}

bool cpu_enable_avx_fma(bool enable)
{
    if (enable)
        supports_fma = _supports_fma();
    else
        supports_fma = false;

    return supports_fma;
}

} // namespace detail

namespace iir
{
void convolve_iir_inner_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs)
{
    if (detail::cpu_has_avx2())
        convolve_iir_inner_single_avx(input, n_pixels, n_times, output, coefs);
    else
        convolve_iir_inner_single_noavx(input, n_pixels, n_times, output, coefs);
}

void convolve_iir_outer_single(const float *input, const unsigned int n_pixels, const unsigned n_times, float *output,
                               const Coefficients &coefs)
{
    if (detail::cpu_has_avx2())
        convolve_iir_outer_single_avx(input, n_pixels, n_times, output, coefs);
    else
        convolve_iir_outer_single_noavx(input, n_pixels, n_times, output, coefs, n_times);
}

} // namepsace iir

} // namespace fastfilters
