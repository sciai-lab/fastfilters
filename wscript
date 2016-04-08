from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'


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

	cfg.check_cpufeatures(['avx', 'fma'])
	cfg.check_cpuid()
	cfg.check_xgetbv()

	cfg.load('python')
	cfg.check_python_version((2,3))
	cfg.check_python_headers()

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include'])
	cfg.env.append_value('CFLAGS', ['-std=c99'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11'])

	cfg.write_config_header('include/config.h')

def build(bld):
	sources = ["src/cpu.c", "src/fastfilters.c", "src/memory.c"]
	sources_python = ["src/bindings_python.cxx"]


	bld.objects(
		source  = sources,
		target  = 'objs_shlib',
		uselib  = 'cshlib')
	bld.objects(
		source  = sources,
		target  = 'objs_stlib',
		uselib  = 'cstlib')

	bld.shlib(features='c', source=["src/dummy.c"], target='fastfilters', use="objs_shlib", name="fastfilters_shared")
	bld(features='c cstlib', source=["src/dummy.c"], target='fastfilters', use="objs_stlib", name="fastfilters_static")
	bld.shlib(features='pyext', source=sources_python, target='fastfilters', use="objs_shlib", name="fastfilters_pyext")
