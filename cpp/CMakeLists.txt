cmake_minimum_required(VERSION 3.22)
project(FUSINTER_cpp)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(external/eigen)
add_subdirectory(external/Catch2)
add_subdirectory(external/pybind11)

add_library(FUSINTER_cpp lib/library.cpp paper_data.h lib/TableManager.h lib/Splitter.h lib/typedefs.h lib/errors.h lib/MergeValueComputer.h lib/fusinter.h)
target_include_directories(FUSINTER_cpp PUBLIC external/eigen)

add_executable(TEST lib/tests.cpp paper_data.cpp lib/MergeValueComputer.h lib/fusinter.h)
target_include_directories(TEST PUBLIC external/eigen lib)

target_link_libraries(TEST PRIVATE Catch2::Catch2WithMain)
target_include_directories(TEST PUBLIC external/Catch2/src)

add_executable(MAIN main.cpp paper_data.cpp lib/MergeValueComputer.h lib/fusinter.h)
target_link_libraries(MAIN PRIVATE FUSINTER_cpp)

set(Python_VIRTUALENV FIRST)

pybind11_add_module(FUSINTER_cpp_pybind pybind11.cpp)
target_include_directories(FUSINTER_cpp_pybind PUBLIC external/eigen)
