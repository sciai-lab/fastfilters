from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'

AVXTEST = '''#include <immintrin.h>
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

@conf
def check_avx2(self):
	flags = ["/arch:AVX2", "-mavx -mavx2 -mfma"]
	msg='Testing AVX2/FMA support'

	self.start_msg('Checking AVX2/FMA compiler support')
	for flag in flags:
		try:
			self.check(header_name='immintrin.h', define_name='HAVE_AVX2_FMA', msg=msg,
				fragment=AVXTEST, errmsg='Compiler does not support AVX2/FMA instruction set.',
				features='cxx', cxxflags=flag, uselib_store="AVX2_FMA")
		except self.errors.ConfigurationError:
			continue
		else:
			self.end_msg("yes (%s)" % flag)
			return flag
	self.fatal('Compiler does not support AVX2/FMA instruction set.')

@conf
def check_cpu_avx2(self):
	self.start_msg('Checking AVX2/FMA CPU support')
	try:
		self.check(header_name='immintrin.h', msg='',
			fragment=AVXTEST, errmsg='CPU does not support AVX2/FMA instruction set.',
			features='cxx cxxprogram', cxxflags=self.env.CXXFLAGS_AVX2_FMA, uselib_store="AVX2_FMA_CPU",
			execute=True)
	except self.errors.ConfigurationError:
		self.end_msg("no. AVX tests will be skipped.", color='RED')
	else:
		self.end_msg("yes.")

@conf
def check_cxx11(self):
	self.check(msg='Checking for C++11 support', errmsg='Compiler does not support C++11.', features='cxx', cxxflags='-std=c++11')


def options(opt):
	opt.load('compiler_cxx')
	opt.load('python')
	opt.load('waf_unit_test')

	opt.add_option('--disable-python', help=("Don't build python bindings."), action='store_true', default=False, dest='python_disable')
	opt.add_option('--disable-tests', help=("Don't run tests."), action='store_true', default=False, dest='tests_disable')

def configure(cfg):
	cfg.load('compiler_cxx')
	cfg.load('waf_unit_test')

	cfg.check_cxx11()
	cfg.check_avx2()
	cfg.check_cpu_avx2()

	if not cfg.options.python_disable:
		cfg.load('python')
		cfg.check_python_version((2,3))
		cfg.check_python_headers()

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11', '-W', '-Wall', '-O3'])

	cfg.env.python_disable = cfg.options.python_disable
	cfg.env.tests_disable = cfg.options.tests_disable

def build(bld):
	sources_noavx = ["src/avx.cxx", "src/convolve_fir.cxx", "src/convolve_iir.cxx", "src/convolve_iir_deriche.cxx"]
	sources_avx = ["src/convolve_iir_avx.cxx", "src/convolve_fir_avx.cxx"]
	sources_python = ["src/pybind.cxx"]

	TaskGen.declare_chain(
						name      = 'copy_bin',
						rule      = "cp ${SRC} ${TGT}",
						ext_in    = '.bin',
						ext_out   = '.b',
						before    = 'utest',
						reentrant = False)

	bld.objects(
		source  = sources_noavx,
		target  = 'objs_noavx',
		uselib  = 'cxxshlib')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_avx',
		cxxflags = bld.env.CXXFLAGS_AVX2_FMA,
		uselib  = 'cxxshlib')

	bld.shlib(features='cxx', source=["src/dummy.cxx"], target='fastfilters', use="objs_avx objs_noavx")

	if not bld.env.python_disable:
		bld.shlib(features='pyext', source=sources_python, target='pyfastfilters', use="fastfilters")

	if not bld.env.tests_disable:
		tests_common = bld.path.ant_glob("tests/*.bin")

		tests = ["iir.cxx"]
		tests_avx = []

		for test in tests:
			bld.program(features='cxx test', source=["tests/" + test] + tests_common, target="test_" + test[:-4], use="fastfilters")


		if 'CXXFLAGS_AVX2_FMA_CPU' in bld.env.keys():
			for test in tests_avx:
				bld.program(features='cxx test', source=["tests/" + test] + tests_common, target="test_" + test[:-4], use="fastfilters")

		bld.options.all_tests = True
		bld.add_post_fun(waf_unit_test.summary)
		bld.add_post_fun(waf_unit_test.set_exit_code)
