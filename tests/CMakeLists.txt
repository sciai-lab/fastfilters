file(GLOB PY_TESTS
    RELATIVE  "${CMAKE_CURRENT_SOURCE_DIR}"
    test_*.py
    )
file(GLOB PY_TESTS_FULLPATH
    test_*.py
    )
add_custom_target(fastfilters_py_test DEPENDS ${PY_TESTS_FULLPATH})
add_dependencies(fastfilters_py_test fastfilters_py)

foreach(test_file ${PY_TESTS})
add_custom_COMMAND(
    TARGET fastfilters_py_test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}" "${CMAKE_CURRENT_BINARY_DIR}/${test_file}"
    COMMENT "Copying Python tests")
endforeach()
if (MSVC)
    add_custom_COMMAND(
        TARGET fastfilters_py_test
        POST_BUILD
        COMMAND if not exist "${CMAKE_CFG_INTDIR}" mkdir "${CMAKE_CFG_INTDIR}")
endif()
add_custom_COMMAND(
    TARGET fastfilters_py_test
    POST_BUILD
    COMMAND python -c "import nose; nose.main()" . "${CMAKE_CFG_INTDIR}" 
    VERBATIM)


if(CMAKE_MAJOR_VERSION LESS 3)
    DEPENDENCY_PATH(FASTFILTERS_PATH fastfilters)
    DEPENDENCY_PATH(FASTFILTERSTEST_PATH fastfilterstest)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/set_paths.py.cmake2.in
                   ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py
                   @ONLY)
else()
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/set_paths.py.in
                   ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py.in
                   @ONLY)

    # two-stage file configuration is necessary because certain target
    # properties are only known at generation time (policy CMP0026)
    if(MSVC)
        file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py
                      INPUT  ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py.in
                      CONDITION $<CONFIG:Release>)
    else()
        file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py
                      INPUT  ${CMAKE_CURRENT_BINARY_DIR}/set_paths.py.in)
    endif()
endif()
