from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'

@conf
def check_cxx11(self):
	self.check(msg='Checking for C++11 support', errmsg='Compiler does not support C++11.', features='cxx', cxxflags='-std=c++11')


TEST_POSIX_MEMALIGN = '''
    #include <stdlib.h>
    int main()
    {
        void *ptr;
        posix_memalign(&ptr, 32, 1024);
        return 0;
    }
'''
@conf
def check_posix_memalign(self):
	self.check_cxx(header_name='stdlib.h', msg='Checking if the \'posix_memalign\' function exists.', mandatory=False, features='cxx', define_name='HAVE_POSIX_MEMALIGN', fragment=TEST_POSIX_MEMALIGN)

TEST_ALIGNED_MALLOC = '''
    #include <malloc.h>
    int main()
    {
        void *ptr = _aligned_malloc(1024, 32);
        _aligned_free(ptr);
        return 0;
    }
'''
@conf
def check_aligned_malloc(self):
	self.check_cxx(header_name='malloc.h', msg='Checking if the \'_aligned_malloc/free\' functions exist', mandatory=False, features='cxx', define_name='HAVE_ALIGNED_MALLOC', fragment=TEST_ALIGNED_MALLOC)


TEST_64BIT = '''
#include  <cassert>

#if _WIN32 || _WIN64
#if _WIN64
#define COMPILED_64BIT
#endif
#endif

#if __GNUC__
#if __x86_64__
#define COMPILED_64BIT
#endif
#endif

int main()
{
#if defined(COMPILED_64BIT)
	assert(sizeof(void*) == 8);
#else
    #error "not 64bit!"
#endif
	return 0;
}
'''
@conf
def check_64bit(self):
	self.check_cxx(msg='Checking if we compile 64bit binaries', mandatory=True, execute=True, fragment=TEST_64BIT)

TEST_CLANG_HOTFIX = '''
#include <stdio.h>
#include <sys/cdefs.h>

extern void foo_alias (void) __asm ("foo");

__extern_always_inline void
foo (void)
{
  puts ("hi oh world!");
  return foo_alias ();
}


void
foo_alias (void)
{
  puts ("hell oh world");
}

int
main ()
{
  foo ();
}
'''
@conf
def check_clang_hotfix(self):
	res = self.check(msg='Checking if we can compile without the __extern_always_inline clang hotfix', mandatory=False, features='cxx', cxxflags=['-Ofast'], fragment=TEST_CLANG_HOTFIX)
	if not res:
		self.env.append_value('CXXFLAGS', '-D__extern_always_inline=extern inline')

@conf
def check_cxx_flag(self, flag):
	res = self.check(msg='Checking if the compiler accepts the \'%s\' flag' % flag, mandatory=False, features='cxx', cxxflags=flag)
	if res:
		self.env.append_value('CXXFLAGS', [flag])

@conf
def check_vigra(self, includes=None):
	self.check_cxx(
		header_name='vigra/config_version.hxx',
		fragment='''
			#include <vigra/config_version.hxx>
			#include <iostream>
			#include <exception>
			#include <stdexcept>
			int main()
			{
				#if VIGRA_VERSION_MAJOR == 1 && VIGRA_VERSION_MINOR >= 11
				std::cout << VIGRA_VERSION;
				#else
				throw std::runtime_error("Unsupported vigra version.");
				#endif
			}
		''',
		execute=True,
		uselib_store='vigra',
		msg='Checking for VIGRA',
		define_name='HAVE_VIGRA',
		includes=includes,
		use='vigra',
		var='have_vigra'
		)


def waf_unit_test_summary(bld):
	"""
	Display an execution summary::
		def build(bld):
			bld(features='cxx cxxprogram test', source='main.c', target='app')
			from waflib.Tools import waf_unit_test
			bld.add_post_fun(waf_unit_test.summary)
	"""
	lst = getattr(bld, 'utest_results', [])
	if lst:
		Logs.pprint('CYAN', 'execution summary')

		total = len(lst)
		tfail = len([x for x in lst if x[1]])

		Logs.pprint('CYAN', '  tests that pass %d/%d' % (total-tfail, total))
		for (f, code, out, err) in lst:
			if not code:
				Logs.pprint('CYAN', '    %s' % f)
				if out:
					Logs.pprint('CYAN', '      stdout:')
					Logs.pprint('NORMAL', '\n'.join(['        %s'%v for v in out.decode('utf-8').splitlines()]))
				if err:
					Logs.pprint('CYAN', '      stderr:')
					Logs.pprint('NORMAL', '\n'.join(['        %s'%v for v in err.decode('utf-8').splitlines()]))

		Logs.pprint('CYAN', '  tests that fail %d/%d' % (tfail, total))
		for (f, code, out, err) in lst:
			if code:
				Logs.pprint('CYAN', '    %s' % f)
				if out:
					Logs.pprint('RED', '      stdout:')
					Logs.pprint('RED', '\n'.join(['        %s'%v for v in out.decode('utf-8').splitlines()]))
				if err:
					Logs.pprint('RED', '      stderr:')
					Logs.pprint('RED', '\n'.join(['        %s'%v for v in err.decode('utf-8').splitlines()]))

def options(opt):
	opt.load('python')
	opt.load('waf_unit_test')
	opt.load('compiler_cxx')
	opt.load('cpufeatures', tooldir='waftools')

	opt.add_option('--gxx', help=("Prefer g++ as compiler"), action='store_true', default=False, dest='gxx')
	opt.add_option('--clang', help=("Prefer clang++ as compiler"), action='store_true', default=False, dest='clang')

	opt.add_option('--disable-python', help=("Don't build python bindings."), action='store_true', default=False, dest='python_disable')
	opt.add_option('--disable-tests', help=("Don't run tests."), action='store_true', default=False, dest='tests_disable')

	opt.add_option('--vigra-includes', help=("VIGRA include directory for performance tests."), action='store', default='', dest='vigra_incdir')
	opt.add_option('--without-vigra', help=("Disable VIGRA tests."), action='store_true', default=False, dest='vigra_disable')

	opt.add_option('--without-opencv', help=("Disable OpenCV tests."), action='store_true', default=False, dest='opencv_disable')

	opt.add_option('--debug', help=("Compile with debug symbols."), action='store_true', default=False, dest='enable_debug')

def configure(cfg):
	if cfg.options.clang:
		cfg.load('clangxx')
	elif cfg.options.gxx:
		cfg.load('gxx')

	cfg.load('compiler_cxx')

	cfg.load('waf_unit_test')
	cfg.load('cpufeatures', tooldir='waftools')

	cfg.check_cxx11()
	cfg.check_64bit()
	cfg.check_library(mode='cxx')

	cfg.check_cpufeatures(['avx', 'fma'])
	cfg.check_cpuid()
	cfg.check_xgetbv()

	cfg.check_posix_memalign()
	cfg.check_aligned_malloc()



	if 'msvc' in (cfg.env.CC_NAME, cfg.env.CXX_NAME):
		cfg.check_cxx_flag("/O2")
		cfg.check_cxx_flag("/W4")
		cfg.check_cxx_flag('/EHsc')
		cfg.check(msg='Checking if the compiler accepts the \'/fp:fast\' flag', mandatory=True, features='cxx', cxxflags=['-Ofast'], uselib_store="fastopt")
	else:
		cfg.check_cxx_flag("-Wall")
		cfg.check_cxx_flag("-Wextra")
		cfg.check(msg='Checking if the compiler accepts the \'-O3\' flag', mandatory=True, features='cxx', cxxflags=['-O3'], uselib_store="normalopt")
		cfg.check(msg='Checking if the compiler accepts the \'-Ofast\' flag', mandatory=True, features='cxx', cxxflags=['-Ofast'], uselib_store="fastopt")
		cfg.check_clang_hotfix()

	if not cfg.options.vigra_disable:
		cfg.check_vigra([cfg.options.vigra_incdir])

	if not cfg.options.opencv_disable:
		cfg.check_cfg(package='opencv', args='--cflags --libs', uselib_store='opencv', use='opencv')

	if not cfg.options.python_disable:
		cfg.load('python')
		cfg.check_python_version((2,3))
		cfg.check_python_headers()

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11'])
	cfg.env.append_value('CXXFLAGS_cxxshlib', ['-DFASTFILTERS_SHARED_LIBRARY=1'])
	cfg.env.append_value('CXXFLAGS_cxxstlib', ['-DFASTFILTERS_STATIC_LIBRARY=1'])

	if cfg.options.enable_debug:
		cfg.env.append_value("CXXFLAGS", ['-g'])

	cfg.env.python_disable = cfg.options.python_disable
	cfg.env.tests_disable = cfg.options.tests_disable
	cfg.env.vigra_disable = cfg.options.vigra_disable
	cfg.env.opencv_disable = cfg.options.opencv_disable

	cfg.write_config_header('include/config.h')

def build(bld):
	sources_noavx = ["src/cpu.cxx", "src/convolve_fir.cxx", "src/convolve_fir_nosimd.cxx", "src/convolve_iir.cxx", "src/convolve_iir_nosimd.cxx", "src/kernel_iir_deriche.cxx", "src/kernel_fir_gaussian.cxx"]
	sources_avx = ["src/convolve_fir_avx.cxx"]
	sources_avx_fma = ["src/convolve_iir_avx.cxx", "src/convolve_fir_avx.cxx"]
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
		uselib  = 'cxxshlib fastopt')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_avx',
		uselib  = 'cxxshlib cpu_avx fastopt')
	bld.objects(
		source  = sources_avx_fma,
		target  = 'objs_avx_fma',
		uselib  = 'cxxshlib cpu_avx cpu_fma fastopt')

	bld.objects(
		source  = sources_noavx,
		target  = 'objs_st_noavx',
		uselib  = 'cxxstlib fastopt')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_st_avx',
		uselib  = 'cxxstlib cpu_avx fastopt')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_st_avx_fma',
		uselib  = 'cxxstlib cpu_avx cpu_fma fastopt')

	bld.shlib(features='cxx', source=["src/dummy.cxx"], target='fastfilters', use="objs_avx objs_avx_fma objs_noavx", uselib="fastopt", name="fastfilters_shared")

	if 'msvc' in (bld.env.CC_NAME, bld.env.CXX_NAME):
		static_name = 'fastfilters_static'
	else:
		static_name = 'fastfilters'

	bld(features='cxx cxxstlib', source=["src/dummy.cxx"], target=static_name, use="objs_st_avx objs_st_avx_fma objs_st_noavx", uselib="fastopt", name="fastfilters_static")

	if not bld.env.python_disable:
		bld.shlib(features='pyext', source=sources_python, target='fastfilters', use="fastfilters_shared", name="fastfilters_pyext")

	if not bld.env.tests_disable:
		tests_common = bld.path.ant_glob("tests/*.bin")

		tests = ["cpufeatures.cxx", "gaussian.cxx"]
		tests_vigra = ["vigra.cxx"]
		tests_opencv = ["opencv.cxx"]

		for test in tests:
			bld.program(features='cxx test', uselib='normalopt', source=["tests/" + test] + tests_common, target="test_" + test[:-4], use="fastfilters_shared")

		if not bld.env.vigra_disable:
			for test in tests_vigra:
				bld.program(features='cxx test', uselib='vigra normalopt', source=["tests/" + test], target="test_" + test[:-4], use="fastfilters_shared")

		if not bld.env.opencv_disable:
			for test in tests_opencv:
				bld.program(features='cxx test', uselib = 'opencv normalopt', source=["tests/" + test], target="test_" + test[:-4], use="fastfilters_shared")

		bld.options.all_tests = True
		bld.add_post_fun(waf_unit_test_summary)
		bld.add_post_fun(waf_unit_test.set_exit_code)
