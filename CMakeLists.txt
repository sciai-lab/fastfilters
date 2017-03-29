cmake_minimum_required (VERSION 2.8.8)
project (fastfilters)

include(GetGitRevisionDescription.cmake)

git_describe(FF_VERSION --tags)
string(SUBSTRING ${FF_VERSION} 1 -1 FF_VERSION)


set(FF_UNROLL 10)

add_definitions(-DFF_UNROLL=${FF_UNROLL})
include("pybind11/tools/FindPythonLibsNew.cmake")

include(CheckCCompilerFlag)
check_c_compiler_flag("-std=c99" HAS_C99_FLAG)

include(CheckCSourceCompiles)

if (HAS_C99_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
endif()

check_c_source_compiles( "
int main()
{
    for(int i = 0; i < 5; ++i) { }
    return 0;
} " CAN_COMPILE_C99_FEAT)

if(NOT CAN_COMPILE_C99_FEAT)
  message(FATAL_ERROR "Unsupported compiler - fastfilters requires some C99 features!")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Ofast" HAS_OFAST_FLAG)
check_cxx_compiler_flag("-O3" HAS_O2_FLAG)
check_cxx_compiler_flag("-O2" HAS_O3_FLAG)

if(HAS_O3_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
elseif(HAS_O2_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
endif()

if(HAS_OFAST_FLAG)
  set(OFAST_FLAG "-Ofast")
else()
  set(OFAST_FLAG "")
endif()

check_cxx_compiler_flag("-Wall" HAS_WALL_FLAG)
check_cxx_compiler_flag("-Wextra" HAS_WEXTRA_FLAG)

if(HAS_WALL_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

if(HAS_WEXTRA_FLAG)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
endif()

check_cxx_compiler_flag("-std=c++14" HAS_CPP14_FLAG)
check_cxx_compiler_flag("-std=c++11" HAS_CPP11_FLAG)

check_cxx_compiler_flag("-mavx" HAS_AVX_FLAG)
check_cxx_compiler_flag("-mavx2" HAS_AVX2_FLAG)
check_cxx_compiler_flag("-mfma" HAS_FMA_FLAG)

check_cxx_compiler_flag("/arch:AVX" HAS_ARCH_AVX_FLAG)
check_cxx_compiler_flag("/arch:AVX2" HAS_ARCH_AVX2_FLAG)

if (HAS_AVX_FLAG)
  set(AVX_FLAG "-mavx")
elseif(HAS_ARCH_AVX_FLAG)
  set(AVX_FLAG "/arch:AVX -D__AVX__=1")
else()
  set(AVX_FLAG "")
endif()

if (HAS_AVX2_FLAG)
  set(AVX2_FLAG "-mavx2")
elseif(HAS_ARCH_AVX2_FLAG)
  set(AVX2_FLAG "/arch:AVX2 -D__AVX__=1 -D__FMA__=1 -D__AVX2__=1")
else()
  set(AVX2_FLAG "")
endif()

if (HAS_FMA_FLAG)
  set(FMA_FLAG "-mfma")
elseif(HAS_ARCH_AVX2_FLAG)
  set(FMA_FLAG "/arch:AVX -D__AVX__=1 -D__FMA__=1")
else()
  set(FMA_FLAG "")
endif()

if (HAS_CPP14_FLAG)
  set(PYBIND11_CPP_STANDARD -std=c++14)
elseif (HAS_CPP11_FLAG)
  set(PYBIND11_CPP_STANDARD -std=c++11)
else()
  if(NOT DEFINED MSVC_VERSION OR MSVC_VERSION LESS 1800)
    message(FATAL_ERROR "Unsupported compiler -- pybind11 requires C++11 support!")
  endif()
endif()
set(PYBIND11_CPP_STANDARD ${PYBIND11_CPP_STANDARD} CACHE STRING
    "C++ standard flag, e.g. -std=c++11 or -std=c++14. Defaults to latest available.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND11_CPP_STANDARD}")

if (MSVC) #disable specific VS warnings
  set(disabled_warnings "/wd4305 /wd4127 /wd4571 /wd4514 /wd4820 /wd4668 /wd4623 /wd4626 /wd5027 /wd4625 /wd4191 /wd4371 /wd4710 /wd4244 /wd4267 /wd4456")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${disabled_warnings}")
endif()

set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX ${PYTHON_MODULE_PREFIX} CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} CACHE INTERNAL "")

# Build a Python extension module:
# pybind11_add_module(<name> source1 [source2 ...])
#
function(pybind11_add_module target_name)
  add_library(${target_name} MODULE ${ARGN})
  target_include_directories(${target_name} PUBLIC ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

  # The prefix and extension are provided by FindPythonLibsNew.cmake
  set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(${target_name} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

  if(WIN32 OR CYGWIN)
    # Link against the Python shared library on Windows
    target_link_libraries(${target_name} PRIVATE ${PYTHON_LIBRARIES})
  elseif(APPLE)
    # It's quite common to have multiple copies of the same Python version
    # installed on one's system. E.g.: one copy from the OS and another copy
    # that's statically linked into an application like Blender or Maya.
    # If we link our plugin library against the OS Python here and import it
    # into Blender or Maya later on, this will cause segfaults when multiple
    # conflicting Python instances are active at the same time (even when they
    # are of the same version).

    # Windows is not affected by this issue since it handles DLL imports
    # differently. The solution for Linux and Mac OS is simple: we just don't
    # link against the Python library. The resulting shared library will have
    # missing symbols, but that's perfectly fine -- they will be resolved at
    # import time.

    target_link_libraries(${target_name} PRIVATE "-undefined dynamic_lookup")
  endif()

  if(NOT MSVC)
    # Make sure C++11/14 are enabled
    target_compile_options(${target_name} PUBLIC ${PYBIND11_CPP_STANDARD})

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      # Check for Link Time Optimization support (GCC/Clang)
      check_cxx_compiler_flag("-flto" HAS_LTO_FLAG)
      if(HAS_LTO_FLAG AND NOT CYGWIN)
        target_compile_options(${target_name} PRIVATE -flto)
      endif()

      # Intel equivalent to LTO is called IPO
      if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        check_cxx_compiler_flag("-ipo" HAS_IPO_FLAG)
        if(HAS_IPO_FLAG)
          target_compile_options(${target_name} PRIVATE -ipo)
        endif()
      endif()

      # Default symbol visibility
      target_compile_options(${target_name} PRIVATE "-fvisibility=hidden")

      # Strip unnecessary sections of the binary on Linux/Mac OS
      if(CMAKE_STRIP)
        if(APPLE)
          add_custom_command(TARGET ${target_name} POST_BUILD
                             COMMAND ${CMAKE_STRIP} -u -r $<TARGET_FILE:${target_name}>)
        else()
          add_custom_command(TARGET ${target_name} POST_BUILD
                             COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${target_name}>)
        endif()
      endif()
    endif()
  elseif(MSVC)
    # /MP enables multithreaded builds (relevant when there are many files), /bigobj is
    # needed for bigger binding projects due to the limit to 64k addressable sections
    target_compile_options(${target_name} PRIVATE /MP /bigobj)

    # Enforce link time code generation on MSVC, except in debug mode
    target_compile_options(${target_name} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/GL>)

    # Fancy generator expressions don't work with linker flags, for reasons unknown
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY LINK_FLAGS_RELEASE /LTCG)
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY LINK_FLAGS_MINSIZEREL /LTCG)
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO /LTCG)
  endif()
endfunction()

# Compile with compiler warnings turned on
function(pybind11_enable_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wconversion)
  endif()
endfunction()


include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS_OLD "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAG}")

check_cxx_source_compiles( "#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
int main()
{
    __m256 a = _mm256_set1_ps(rand());
    __m256 b = _mm256_set1_ps(rand());
    b = _mm256_add_ps(a, b);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf(\"%f\", result);
    return 0;
} " CAN_COMPILE_AVX)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_OLD}")

set(CMAKE_REQUIRED_FLAGS_OLD "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS} ${AVX_FLAG} ${FMA_FLAG}")

check_cxx_source_compiles( "#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
int main()
{
    __m256 a = _mm256_set1_ps(rand());
    __m256 b = _mm256_set1_ps(rand());
    b = _mm256_fmadd_ps(a, a, b);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf(\"%f\", result);
    return 0;
} " CAN_COMPILE_FMA)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_OLD}")

set(CMAKE_REQUIRED_FLAGS_OLD "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS} ${AVX2_FLAG}")

check_cxx_source_compiles( "
    #include <immintrin.h>
    #include <stdlib.h>
    #include <stdio.h>
    int main()
    {
    float test = 1.0;
    __m128 a = _mm_setzero_ps();
    __m256 b = _mm256_broadcastss_ps(a);
    float result = _mm_cvtss_f32(_mm256_extractf128_ps(b, 0));
    printf(\"%f\", result);
    return 0;
    }" CAN_COMPILE_AVX2)

set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_OLD}")

function(check_cpu_supports flagname defname)
    check_cxx_source_compiles( "#include <stdio.h> \n int main() { return __builtin_cpu_supports(\"${flagname}\"); }" ${defname})
endfunction()

check_cpu_supports("avx" "HAVE_GNU_CPU_SUPPORTS_AVX")
check_cpu_supports("avx2" "HAVE_GNU_CPU_SUPPORTS_AVX2")
check_cpu_supports("fma" "HAVE_GNU_CPU_SUPPORTS_FMA")


check_cxx_source_compiles( "
    #include <stdio.h>
    int main()
    {
        unsigned int a, b, c, d;

        a = 1;
        c = 0;
        __asm__ __volatile__ (\"cpuid\" : \"+a\"(a), \"+b\"(b), \"+c\"(c), \"=d\"(d));
        return a;
    }" HAVE_ASM_CPUID)

check_cxx_source_compiles("
    #include <stdio.h>
    #include <intrin.h>
    int main()
    {
        int cpuid[4];
        __cpuidex(cpuid, 7, 0);
        return cpuid[0];
    }" HAVE_CPUIDEX)

check_cxx_source_compiles("
    #include <stdio.h>
    #include <intrin.h>
    int main()
    {
        unsigned int xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        return xcr0;
    }" HAVE_INTRIN_XGETBV)



check_cxx_source_compiles( "
    #include <stdio.h>
    int main()
    {
        unsigned int a, b, c;

        c = 0;
        __asm__ __volatile__(\"xgetbv\" : \"=a\"(a), \"=d\"(b) : \"c\"(c));
        return a;
    }" HAVE_ASM_XGETBV)

check_cxx_source_compiles( "
        #include <stdbool.h>
        int main(int argc, char *argv[])
        {
            (void)argv;
            return __builtin_expect(argc > 2, false);
        }" HAVE_BUILTIN_EXPECT)

include(CheckIncludeFiles)
check_include_files(cpuid.h HAVE_CPUID_H)

if(NOT CAN_COMPILE_AVX)
    message( FATAL_ERROR "Compiler cannot emit avx instructions.")
endif(NOT CAN_COMPILE_AVX)
if(NOT CAN_COMPILE_AVX2)
    message( FATAL_ERROR "Compiler cannot emit avx2 instructions.")
endif(NOT CAN_COMPILE_AVX2)
if(NOT CAN_COMPILE_FMA)
    message( FATAL_ERROR "Compiler cannot emit fma instructions.")
endif(NOT CAN_COMPILE_FMA)

configure_file (
  "${PROJECT_SOURCE_DIR}/src/library/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
  )

include_directories("${PROJECT_SOURCE_DIR}/src/library/")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/pybind11/include")
include_directories("${PROJECT_SOURCE_DIR}/boost-preprocessor/include")
include_directories("${PROJECT_BINARY_DIR}")

configure_file(${PROJECT_SOURCE_DIR}/src/library/linalg_avx2.c ${PROJECT_BINARY_DIR}/linalg_avx2.avx.c COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/src/library/linalg_avx2.c ${PROJECT_BINARY_DIR}/linalg_avx2.avx2.c COPYONLY)

set_source_files_properties(${PROJECT_SOURCE_DIR}/src/library/linalg_avx.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${OFAST_FLAG}")
set_source_files_properties(${PROJECT_BINARY_DIR}/linalg_avx2.avx.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${OFAST_FLAG}")
set_source_files_properties(${PROJECT_BINARY_DIR}/linalg_avx2.avx2.c PROPERTIES COMPILE_FLAGS "${AVX2_FLAG} ${OFAST_FLAG}")

configure_file(${PROJECT_SOURCE_DIR}/src/library/fir_convolve_avx.c ${PROJECT_BINARY_DIR}/fir_convolve_avx.avx.c COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/src/library/fir_convolve_avx.c ${PROJECT_BINARY_DIR}/fir_convolve_avx.avxfma.c COPYONLY)

set_source_files_properties(${PROJECT_BINARY_DIR}/fir_convolve_avx.avx.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${OFAST_FLAG}")
set_source_files_properties(${PROJECT_BINARY_DIR}/fir_convolve_avx.avxfma.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${FMA_FLAG} ${OFAST_FLAG}")

set(number ${FF_UNROLL})
set(copied_files "")
while( number GREATER 0 )
  configure_file(${PROJECT_SOURCE_DIR}/src/library/fir_convolve_avx_impl.c ${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avx.c COPYONLY)
  set_source_files_properties(${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avx.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${OFAST_FLAG} -DFF_KERNEL_LEN=${number}")

  configure_file(${PROJECT_SOURCE_DIR}/src/library/fir_convolve_avx_impl.c ${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avxfma.c COPYONLY)
  set_source_files_properties(${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avxfma.c PROPERTIES COMPILE_FLAGS "${AVX_FLAG} ${OFAST_FLAG} ${FMA_FLAG} -DFF_KERNEL_LEN=${number}")

  set(copied_files ${copied_files} ${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avx.c ${PROJECT_BINARY_DIR}/fir_convolve_avx_impl.${number}.avxfma.c)

  math( EXPR number "${number} - 1" ) # decrement number
endwhile( number GREATER 0 )

add_library(fastfilters SHARED src/library/array.c
src/library/cpu.c
src/library/dummy.c
src/library/fastfilters.c
src/library/fir_convolve.c
src/library/fir_convolve_nosimd.c
src/library/fir_filters.c
src/library/fir_kernel.c
${PROJECT_BINARY_DIR}/linalg_avx2.avx.c
${PROJECT_BINARY_DIR}/linalg_avx2.avx2.c
src/library/linalg_avx.c
src/library/linalg.c
src/library/memory.c
${PROJECT_BINARY_DIR}/fir_convolve_avx.avx.c
${PROJECT_BINARY_DIR}/fir_convolve_avx.avxfma.c
${copied_files})

target_compile_definitions(fastfilters PRIVATE FASTFILTERS_SHARED_LIBRARY)
set_target_properties(fastfilters PROPERTIES SOVERSION ${FF_VERSION})

pybind11_add_module(core src/python/core.cxx)
pybind11_enable_warnings (core)
target_link_libraries(core PUBLIC fastfilters)

set_source_files_properties(src/python/core.cxx PROPERTIES COMPILE_FLAGS "-DFF_VERSION_STR=\\\"${FF_VERSION}\\\"")

if(NOT DEFINED FF_INSTALL_DIR OR FF_INSTALL_DIR MATCHES "^$")
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import *; print(get_python_lib(1))" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} FF_INSTALL_DIR)
endif()

set(FF_INSTALL_DIR ${FF_INSTALL_DIR} CACHE PATH "install directory for ff python extension." FORCE)
file(RELATIVE_PATH FF_INSTALL_DIR ${CMAKE_INSTALL_PREFIX} ${FF_INSTALL_DIR})

set(fastfilters_py_tmp_dir "${CMAKE_CURRENT_BINARY_DIR}/python")
file(MAKE_DIRECTORY "${fastfilters_py_tmp_dir}")
file(MAKE_DIRECTORY "${fastfilters_py_tmp_dir}/fastfilters")

add_custom_target(fastfilters_py_lib)
SET(PY_SOURCES
    __init__.py
    )
foreach(lib_file ${PY_SOURCES})
add_custom_COMMAND(
    TARGET fastfilters_py_lib
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/src/python/${lib_file}" "${fastfilters_py_tmp_dir}/fastfilters/${lib_file}"
    COMMENT "Copying Python sources to temporary module directory")
endforeach(lib_file)

add_custom_target(fastfilters_py)
add_dependencies(fastfilters_py fastfilters_py_lib core fastfilters)

ADD_CUSTOM_COMMAND(
        TARGET core
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy "$<TARGET_FILE:core>" "${fastfilters_py_tmp_dir}/fastfilters/"
        COMMENT "Copying pyd file to temporary module directory")

install(TARGETS fastfilters ARCHIVE DESTINATION lib RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
install(TARGETS core LIBRARY DESTINATION ${FF_INSTALL_DIR}/fastfilters/)
install(FILES ${PROJECT_SOURCE_DIR}/src/python/__init__.py DESTINATION ${FF_INSTALL_DIR}/fastfilters/)
install(FILES ${PROJECT_SOURCE_DIR}/include/fastfilters.h DESTINATION include)

ADD_SUBDIRECTORY(tests)
