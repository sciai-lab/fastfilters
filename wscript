from waflib import Task, Options, Configure, TaskGen, Logs, Build, Utils, Errors
from waflib.TaskGen import feature, before_method
from waflib.Configure import conf
from waflib.Tools import waf_unit_test
from waflib.Build import BuildContext

VERSION='0.0.1'
APPNAME='libfastfilters'

srcdir = 'src'
blddir = 'build'

FF_UNROLL = 10

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


@conf
def check_clang_hotfix(self):
	res = self.check(msg='Checking if we can compile without the __extern_always_inline clang hotfix', mandatory=False, features='cxx', cxxflags=['-Ofast'], fragment='''
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
''')
	if not res:
		self.env.append_value('CXXFLAGS', '-D__extern_always_inline=extern inline')
		self.env.append_value('CFLAGS', '-D__extern_always_inline=extern inline')

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
	cfg.load('run_py_script', tooldir='waftools')

	#cfg.check_library(mode='c')

	cfg.check_clang_hotfix()
	cfg.check_builtin_expect()
	cfg.check_cpufeatures(['avx', 'fma'])
	cfg.check_cpuid()
	cfg.check_xgetbv()

	cfg.load('python')
	cfg.check_python_version((2,3))
	cfg.check_python_headers()

	cfg.check(features='c cprogram', lib=['m'], cflags=['-Wall'], uselib_store='M')

	cfg.env.append_value('INCLUDES', ['pybind11/include', 'include', 'src', 'boost-preprocessor/include'])
	cfg.env.append_value('CFLAGS', ['-std=c99', '-Wextra', '-Wall', '-funroll-loops', '-ffast-math', '-fvisibility=hidden'])
	cfg.env.append_value('CXXFLAGS', ['-std=c++11', '-Wextra', '-Wall', '-fvisibility=hidden'])

	if cfg.options.enable_debug:
		cfg.env.append_value('CFLAGS', ['-Og', '-g'])
		cfg.env.append_value('CXXFLAGS', ['-Og', '-g'])
	else:
		cfg.env.append_value('CFLAGS', ['-Ofast'])
		cfg.env.append_value('CXXFLAGS', ['-Ofast'])

	cfg.env.append_value('CFLAGS_cshlib', ['-DFASTFILTERS_SHARED_LIBRARY=1'])
	cfg.env.append_value('CFLAGS_cstlib', ['-DFASTFILTERS_STATIC_LIBRARY=1'])

	cfg.define('FF_UNROLL', FF_UNROLL)

	cfg.write_config_header('include/config.h')

def build(bld):
	sources = ["src/cpu.c", "src/fastfilters.c", "src/memory.c", "src/fir_kernel.c", "src/linalg.c", "src/fir_convolve_nosimd.c", "src/fir_convolve.c", "src/fir_filters.c", "src/array.c"]
	sources_avx = ["src/fir_convolve_avx.c"]
	sources_avx_fma = ["src/fir_convolve_avx.c", "src/linalg_avx.c"]
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
		source  = sources_avx_fma,
		target  = 'objs_shlib_avx_fma',
		uselib  = 'cshlib cpu_avx cpu_fma')

	bld.objects(
		source  = sources,
		target  = 'objs_stlib',
		uselib  = 'cstlib')
	bld.objects(
		source  = sources_avx,
		target  = 'objs_stlib_avx',
		uselib  = 'cstlib cpu_avx')
	bld.objects(
		source  = sources_avx_fma,
		target  = 'objs_stlib_avx_fma',
		uselib  = 'cstlib cpu_avx cpu_fma',
		msg = 'lol')

	avx_use_st = []
	avx_use_sh = []
	for i in range(1,FF_UNROLL):
		tname = 'objs_avx_st_%d' % i
		bld.objects(source="src/fir_convolve_avx_impl.c", target=tname, uselib='cstlib cpu_avx', cflags='-DFF_KERNEL_LEN=%d' % i)
		avx_use_st.append(tname)

		tname = 'objs_avx_sh_%d' % i
		bld.objects(source="src/fir_convolve_avx_impl.c", target=tname, uselib='cshlib cpu_avx', cflags='-DFF_KERNEL_LEN=%d' % i)
		avx_use_sh.append(tname)

		tname = 'objs_avxfma_st_%d' % i
		bld.objects(source="src/fir_convolve_avx_impl.c", target=tname, uselib='cstlib cpu_avx cpu_fma', cflags='-DFF_KERNEL_LEN=%d' % i)
		avx_use_st.append(tname)

		tname = 'objs_avxfma_sh_%d' % i
		bld.objects(source="src/fir_convolve_avx_impl.c", target=tname, uselib='cshlib cpu_avx cpu_fma', cflags='-DFF_KERNEL_LEN=%d' % i)
		avx_use_sh.append(tname)

	avx_use_sh = ' '.join(avx_use_sh)
	avx_use_st = ' '.join(avx_use_st)

	shlib = bld.shlib(features='c', source=["src/dummy.c"], target='fastfilters', use="objs_shlib objs_shlib_avx objs_shlib_avx_fma " + avx_use_sh, name="fastfilters_shared")
	bld(features='c cstlib', source=["src/dummy.c"], target='fastfilters', use="objs_stlib objs_stlib_avx objs_stlib_avx_fma " + avx_use_st, name="fastfilters_static")
	pyext = bld.shlib(features='pyext', source=sources_python, target='fastfilters', use="fastfilters_shared", name="fastfilters_pyext")


	pyext.post()
	shlib.post()

	global pyext_target, shlib_target
	pyext_target = pyext.tasks[-1].outputs[0]
	shlib_target = shlib.tasks[-1].outputs[0]

class debug(BuildContext):
        cmd = 'testimpl'
        fun = 'testimpl'

def testimpl(bld):
	global pyext_target, shlib_target

	if 'pyext_target' not in globals() or 'shlib_target' not in globals(): raise Errors.WafError("Don't run testimpl directly.")

	bdir = bld.path.find_dir('build').abspath()
	tests = bld.path.ant_glob('tests/*.py')

	for test in tests:
		bld(features='run_py_script', source=test.relpath(), target=['%s.b' % test.relpath()], add_to_pythonpath=bdir, add_to_ldpath=bdir, deps=[pyext_target.relpath(), test.relpath()])

def test(ctx):
	Options.commands = ['build', 'testimpl'] + Options.commands
