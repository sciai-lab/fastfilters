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

#if defined(HAVE_XGETBV) || defined(HAVE_CPUIDEX)
#include <intrin.h>
#endif

namespace fastfilters
{

namespace detail
{

enum avx_status_t { AVX_STATUS_UNKNOWN = 0, AVX_STATUS_UNSUPPORTED, AVX_STATUS_SUPPORTED };

static avx_status_t avx_status = AVX_STATUS_UNKNOWN;

#if defined(HAVE_GNU_CPU_SUPPORTS)

static inline bool internal_cpu_has_avx2()
{
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return true;
    else
        return false;
}

#elif defined(HAVE_XGETBV) && defined(HAVE_CPUIDEX)

static inline bool internal_cpu_has_avx2()
{
    // check for CPU AVX2 support: CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1
    unsigned int avxflag;
    // check for CPU FMA support: CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1
    unsigned int fmaflag;

    int cpuid[4];
    __cpuidex(cpuid, 7, 0);
    avxflag = cpuid[1];
    __cpuidex(cpuid, 1, 0);
    fmaflag = cpuid[2];

    if ((avxflag & (1 << 5)) != (1 << 5))
        return false;
    if ((fmaflag & (1 << 12)) != (1 << 12))
        return false;

    unsigned int xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    if (((xcr0 & 6) != 6))
        return false;
    return true;
}

#elif defined(HAVE_INLINE_ASM)

static inline bool internal_cpu_has_avx2()
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    // check for CPU AVX2 support: CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1
    unsigned int avxflag;
    eax = 7;
    ecx = 0;
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    avxflag = ebx;
    if ((avxflag & (1 << 5)) != (1 << 5))
        return false;

    // check for CPU FMA support: CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1
    unsigned int fmaflag;
    eax = 1;
    ecx = 0;
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    fmaflag = ecx;
    if ((fmaflag & (1 << 12)) != (1 << 12))
        return false;

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    unsigned int xcr0;
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
    if (((xcr0 & 6) != 6))
        return false;
    return true;
}
#else
#error "No known way to test for runtime AVX2 support!"
#endif

bool cpu_has_avx2()
{
    if (avx_status == AVX_STATUS_UNKNOWN) {
        if (internal_cpu_has_avx2())
            avx_status = AVX_STATUS_SUPPORTED;
        else
            avx_status = AVX_STATUS_UNSUPPORTED;
    }

    if (avx_status == AVX_STATUS_SUPPORTED)
        return true;
    else
        return false;
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
