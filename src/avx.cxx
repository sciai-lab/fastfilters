#include "fastfilters.hxx"

namespace fastfilters
{

namespace detail
{

enum avx_status_t { AVX_STATUS_UNKNOWN = 0, AVX_STATUS_UNSUPPORTED, AVX_STATUS_SUPPORTED };

static avx_status_t avx_status = AVX_STATUS_UNKNOWN;

static bool internal_cpu_has_avx2()
{
// safeguard because we're not allowed to clobber EBX under 32-bit PIC
// but this code should never be compiled in 32bit mode anyways
#if defined(__i386__)
#error "avx code compiled in 32bit mode"
    return false;
#else

#if defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 8
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return true;
    else
        return false;
#else
    // check for CPU AVX2 support: CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1
    unsigned int avxflag;

    // check for CPU FMA support: CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1
    unsigned int fmaflag;

#if defined(_MSC_VER)
    int cpuid[4];
    __cpuidex(cpuid, 7, 0);
    avxflag = cpuid[1];
    __cpuidex(cpuid, 1, 0);
    fmaflag = cpuid[2];
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    eax = 7;
    ecx = 0;
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    avxflag = ebx;

    eax = 1;
    ecx = 0;
    __asm__("cpuid" : "+b"(ebx), "+a"(eax), "+c"(ecx), "=d"(edx));
    fmaflag = ecx;
#endif // !defined(_MSC_VER)

    if ((avxflag & (1 << 5)) != (1 << 5))
        return false;
    if ((fmaflag & (1 << 12)) != (1 << 12))
        return false;

    // check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
    unsigned int xcr0;

#if defined(_MSC_VER)
    xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif

    if (((xcr0 & 6) != 6))
        return false;
    return true;
#endif // !(defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >=
// 8)

#endif // defined(__i386__)
}

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