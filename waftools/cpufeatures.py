from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf

#print(dir(Options))

features = ["avx", "avx2", "fma"]

flags_gcc = {
	"avx": "-mavx",
	"avx2": "-mavx2",
	"fma": "-mfma"
}

flags_msvc = {
	"avx": "/arch:AVX -D__AVX__=1",
	"avx2": "/arch:AVX2 -D__AVX__=1 -D__FMA__=1 -D__AVX2__=1",
	"fma": "/arch:AVX2 -D__AVX__=1 -D__FMA__=1 -D__AVX2__=1"
}

compiler_mapping = {
	"gcc": flags_gcc,
	"g++": flags_gcc,
	"msvc": flags_msvc,
	"clang": flags_gcc,
	"clang++": flags_gcc
}


TEST_AVX = '''#include <immintrin.h>
    #include <stdlib.h>
    #include <stdio.h>
    int main()
    {
    __m256 a = _mm256_set1_ps(rand());
    __m256 b = _mm256_set1_ps(rand());
    b = _mm256_add_ps(a, b);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf("%f\\n", result);
    return 0;
    }
'''

TEST_FMA = '''#include <immintrin.h>
    #include <stdlib.h>
    #include <stdio.h>
    int main()
    {
    __m256 a = _mm256_set1_ps(rand());
    __m256 b = _mm256_set1_ps(rand());
    b = _mm256_fmadd_ps(a, a, b);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf("%f\\n", result);
    return 0;
    }
'''

TEST_AVX2 = '''#include <immintrin.h>
    #include <stdlib.h>
    #include <stdio.h>
    int main()
    {
    float test = 1.0;
    __m128 a = _mm_setzero_ps();
    __m256 b = _mm256_broadcastss_ps(a);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf("%f\\n", result);
    return 0;
    }
'''

tests = {
	"avx": TEST_AVX,
	"fma": TEST_FMA,
	"avx2": TEST_AVX2
}


TEST_BUILTIN_CPU_SUPPORTS = '''
    #include <stdio.h>
    int main()
    {
        return __builtin_cpu_supports("%s");
    }'''

TEST_CPUID_H = '''
	#include <cpuid.h>
	int main(int argc, char *argv[]) {
		(void)argc; (void)argv;
		unsigned int tmp;
		__get_cpuid(0, &tmp, &tmp, &tmp, &tmp);
		return 0;
	}
'''


TEST_CPUIDEX = '''
    #include <stdio.h>
    #include <intrin.h>
    int main()
    {
        int cpuid[4];
        __cpuidex(cpuid, 7, 0);
        return cpuid[0];
    }
'''

TEST_ASM_CPUID = '''
    #include <stdio.h>
    int main()
    {
	    unsigned int a, b, c, d;

		a = 1;
		c = 0;
		__asm__ __volatile__ ("cpuid" : "+a"(a), "+b"(b), "+c"(c), "=d"(d));
        return a;
    }
'''

TEST_ASM_XGETBV = '''
    #include <stdio.h>
    int main()
    {
	    unsigned int a, b, c;

		c = 0;
		__asm__ __volatile__("xgetbv" : "=a"(a), "=d"(b) : "c"(c));
        return a;
    }
'''

TEST_INTR_XGETBV = '''
    #include <stdio.h>
    #include <intrin.h>
    int main()
    {
        unsigned int xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return xcr0;
    }
'''


@conf
def check_builtin_cpu_supports_flag(self, flag):
	self.check_cc(
		msg='Checking if the compiler supports \'__builtin_cpu_supports(%s)\'' % flag,
		mandatory=False,
		define_name='HAVE_GNU_CPU_SUPPORTS_%s' % flag.upper(),
		fragment=TEST_BUILTIN_CPU_SUPPORTS % flag)


def options(opt):
	opt = opt.add_option_group('CPU features')
	for feature in features:
		opt.add_option(
			'--disable-%s' % feature,
			action='append_const',
			const=feature,
			dest='disable_cpufeatures',
			help=("Disable checks for \'%s\' support" % feature)
			)


@conf
def check_cpuid(cfg, mandatory=True):
	res = cfg.check_cc(
		header_name='cpuid.h',
		define_name='HAVE_CPUID_H',
		msg='Checking if the compiler supports the \'__get_cpuid\' intrinsic',
		fragment=TEST_CPUID_H,
		mandatory=False)

	res2 = cfg.check_cc(
		header_name='intrin.h',
		define_name='HAVE_CPUIDEX',
		msg='Checking if the compiler supports the \'__cpuidex\' intrinsic',
		fragment=TEST_CPUIDEX,
		mandatory=False)

	res3 = cfg.check_cc(
		define_name='HAVE_ASM_CPUID',
		msg='Checking if the compiler supports the \'cpuid\' instruction',
		fragment=TEST_ASM_CPUID,
		mandatory=False)

	if mandatory and res == None and res2 == None and res3 == None:
		raise Errors.ConfigurationError('No known compiler intrinsic to read cpuid.')


@conf
def check_xgetbv(cfg, mandatory=True):
	res = cfg.check_cc(
		define_name='HAVE_ASM_XGETBV',
		msg='Checking if the compiler supports the \'xgetbv\' instruction',
		fragment=TEST_ASM_XGETBV,
		mandatory=False)

	res2 = cfg.check_cc(
		define_name='HAVE_INTRIN_XGETBV',
		msg='Checking if the compiler supports the \'_xgetbv\' intrinsic',
		fragment=TEST_INTR_XGETBV,
		mandatory=False)

	if mandatory and res == None and res2 == None:
		raise Errors.ConfigurationError('No known compiler intrinsic to read xcr0.')

@conf
def check_cpufeatures(cfg, required_compiler_features=[]):
	cc = cfg.env['COMPILER_CC'] or None
	cxx = cfg.env['COMPILER_CXX'] or None

	if not (cc or cxx):
		raise Errors.ConfigurationError("COMPILER_CXX and COMPILER_CC undefined.")

	if cc in compiler_mapping.keys():
		flags = compiler_mapping[cc]
	elif cxx in compiler_mapping.keys():
		flags = compiler_mapping[cxx]
	else:
		raise Errors.ConfigurationError("No cpufeatures support for compilers \'%s\'/\'%s\'" % (cc, cxx))

	if cfg.options.disable_cpufeatures:
		enabled = [f for f in features if f not in cfg.options.disable_cpufeatures]
	else:
		enabled = features

	for flag in required_compiler_features:
		if flag not in enabled:
			raise Errors.ConfigurationError("CPU feature \'%s\' is required but has been disabled by user" % flag)

	for flag in enabled:
		cfg.check_builtin_cpu_supports_flag(flag)
		cfg.check_cc(
				header_name='immintrin.h',
				define_name='HAVE_%s' % flag.upper(),
				msg="Checking if the compiler is able to emit \'%s\' instructions." % flag,
				fragment=tests[flag],
				errmsg='Compiler does not support the \'%s\' instruction set.' % flag,
				uselib_store="cpu_%s" % flag,
				cflags = flags[flag],
				#cxxflags = flags[flag],
				mandatory = flag in required_compiler_features)
