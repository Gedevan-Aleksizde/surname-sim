cmake_minimum_required(VERSION 3.11)

project(surname_sim)
set(CMAKE_CXX_STANDARD 20)
set(PYBIND11_PYTHON_VERSION 3.11)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

find_package(pybind11 REQUIRED)
find_package(GSL REQUIRED)

pybind11_add_module(_sim surname_sim/src/randomsim.cpp)

target_compile_definitions(_sim PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(_sim PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(_sim PRIVATE GSL::gsl GSL::gslcblas)