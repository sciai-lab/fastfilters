from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'

@conf
def check_builtin_expect(cfg):
	cfg.check_cc(
		msg='Checking if the compiler supports \'__builin_expect\'',
		mandatory=False,
		define_name='HAVE_BUILTIN_EXPECT',
		fragment='''
		#include <stdbool.h>
		int main(int argc, char *argv[])
		{
			(void)argv;
			return __builtin_expect(argc > 2, false);
		}
		''')

def options(opt):
	opt.load('python')
	opt.load('compiler_c')
	opt.load('compiler_cxx')
	opt.load('cpufeatures', tooldir='waftools')


	opt.add_option('--debug', help=("Compile with debug symbols."), action='store_true', default=False, dest='enable_debug')

def configure(cfg):
	cfg.load('compiler_c')
	cfg.load('compiler_cxx')

	cfg.load('waf_unit_test')
	cfg.load('cpufeatures', tooldir='waftools')

	cfg.check_library(mode='c')

	cfg.check_builtin_expect()
	cfg.check_cpufeatures(['avx', 'fma'])
	cfg.check_cpuid()
	cfg.check_xgetbv()

	cfg.load('python')
	cfg.check_python_version((2,3))
	cfg.check_python_headers()

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include', 'src', 'boost-preprocessor/include'])
	cfg.env.append_value('CFLAGS', ['-std=c99', '-Wextra', '-Wall', '-Ofast', '-funroll-loops', '-funswitch-loops'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11', '-Wextra', '-Wall', '-Ofast'])

	cfg.env.append_value('CFLAGS_cshlib', ['-DFASTFILTERS_SHARED_LIBRARY=1'])
	cfg.env.append_value('CFLAGS_cstlib', ['-DFASTFILTERS_STATIC_LIBRARY=1'])

	cfg.write_config_header('include/config.h')

def build(bld):
	sources = ["src/cpu.c", "src/fastfilters.c", "src/memory.c", "src/fir_kernel.c", "src/linalg.c", "src/fir_convolve_nosimd.c", "src/fir_convolve.c"]
	sources_avx = ["src/linalg_avx.c"]
	sources_python = ["src/bindings_python.cxx"]


	bld.objects(
		source  = sources,
		target  = 'objs_shlib',
		uselib  = 'cshlib')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_shlib_avx',
		uselib  = 'cshlib cpu_avx')

	bld.objects(
		source  = sources,
		target  = 'objs_stlib',
		uselib  = 'cstlib')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_stlib_avx',
		uselib  = 'cstlib cpu_avx')

	bld.shlib(features='c', source=["src/dummy.c"], target='fastfilters', use="objs_shlib objs_shlib_avx", name="fastfilters_shared")
	bld(features='c cstlib', source=["src/dummy.c"], target='fastfilters', use="objs_stlib objs_stlib_avx", name="fastfilters_static")
	bld.shlib(features='pyext', source=sources_python, target='fastfilters', use="objs_shlib objs_shlib_avx", name="fastfilters_pyext")
