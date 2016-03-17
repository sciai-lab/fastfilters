#include "fastfilters.hxx"

namespace fastfilters
{

namespace detail
{

bool cpu_has_avx2()
{
#if defined(__AVX2__)

// safeguard because we're not allowed to clobber EBX under 32-bit PIC
// but this code should never be compiled in 32bit mode anyways
#if defined(__i386__)
#warning "__AVX2__ defined but compiled in 32bit mode"
	return false;
#else


#if defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 8
	if (__builtin_cpu_supports("avx2"))
		return true;
	else
		return false;
#else
	// check for CPU support: CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1
	unsigned int avxflag;

#if defined(_MSC_VER)
	unsigned int cpuid[4];
	__cpuidex(cpuid, 7, 0);
	avxflag = cpuid[1];
#else
	unsigned int eax, ebx, ecx, edx;
	eax = 7;
	ecx = 0;
	__asm__ ( "cpuid" : "+b" (ebx),"+a" (eax), "+c" (ecx), "=d" (edx));
	avxflag = ebx;
#endif // !defined(_MSC_VER)

	if ((avxflag & (1 << 5)) != (1 << 5))
		return false;

	// check for OS support: XCR0[2] (AVX state) and XCR0[1] (SSE state)
	unsigned int xcr0;
	__asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );

	if (((xcr0 & 6) != 6))
		return false;
	return true;
#endif // !(defined(__GNUC__) && defined(__GNUC_MINOR__) && __GNUC_MINOR__ >= 8)

#endif // defined(__i386__)

#else // defined(__AVX2__)
	return false;
#endif // defined(__AVX2__)
}


} // namespace detail
} // namespace fastfilters