from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'

AVXTEST = '''#include <immintrin.h>
    int main()
    {
    __m256 a = _mm256_setzero_ps();
    __m256 b = _mm256_setzero_ps();
    b = _mm256_fmadd_ps(a, a, b);
    return 0;
    }
'''

@conf
def test_avx2(self):
	flags = ["/arch:AVX2", "-mavx -mavx2 -mfma"]
	msg='Testing AVX2/FMA support'

	self.start_msg('Checking AVX2/FMA support')
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

def options(opt):
	opt.load('compiler_cxx')
	opt.load('python')
	opt.load('waf_unit_test')

	opt.add_option('--disable-python', help=("Don't build python bindings."), action='store_true', default=False, dest='python_disable')
	opt.add_option('--disable-tests', help=("Don't run tests."), action='store_true', default=False, dest='tests_disable')

def configure(cfg):
	cfg.load('compiler_cxx')
	conf.load('waf_unit_test')
	cfg.test_avx2()

	if not Options.options.python_disable:
		cfg.load('python')
		cfg.check_python_version((2,3))
		cfg.check_python_headers()

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11', '-W', '-Wall', '-O3'])

def build(bld):
	src_dir = bld.path.find_dir('src/')
	tests_dir = bld.path.find_dir('tests/')

	sources_noavx = ["src/avx.cxx", "src/convolve_fir.cxx", "src/convolve_iir.cxx", "src/convolve_iir_deriche.cxx"]
	sources_avx = ["src/fastfilters.cxx"]
	sources_python = ["src/pybind.cxx"]

	bld.objects(
		source  = sources_noavx,
		target  = 'objs_noavx',
		uselib  = 'cxxshlib')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_avx',
		cxxflags = bld.env.CXXFLAGS_AVX2_FMA,
		uselib  = 'cxxshlib')

	bld.shlib(features='pyext', source=sources_python, target='fastfilters', use="objs_avx objs_noavx")
	bld.shlib(features='cxx', source=["src/dummy.cxx"], target='fastfilters', use="objs_avx objs_noavx")

	if not Options.options.tests_disable:
		tests = tests_dir.ant_glob("*.cxx")
		for test in tests:
			bld.program(features='cxx test', source=[test], target="test_" + test.name[:-4], use="fastfilters")

		bld.options.all_tests = True
		bld.add_post_fun(waf_unit_test.summary)
