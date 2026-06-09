cmake_minimum_required(VERSION 3.24)

# Pin CUDA host compiler before project().
if(DEFINED ENV{CUDAHOSTCXX} AND NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "$ENV{CUDAHOSTCXX}"
        CACHE PATH "Host compiler for nvcc (gcc12 required by CUDA 12.2)" FORCE)
    message(STATUS "CUDA host compiler pinned to: $ENV{CUDAHOSTCXX}")
endif()

project(membrane
    VERSION 0.1.0
    LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED COMPONENTS
    cudart
    cublas
    cusparse
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(Eigen3 REQUIRED)

find_package(spdlog REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_cmd_result)
    if(pybind11_cmd_result EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_DIR}")
    else()
        message(FATAL_ERROR "pybind11 not found.  Install it:  pip install pybind11")
    endif()
endif()

message(STATUS "pybind11 version : ${pybind11_VERSION}")
message(STATUS "Python3 exe      : ${Python3_EXECUTABLE}")
message(STATUS "CUDA version     : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA host CXX    : ${CMAKE_CUDA_HOST_COMPILER}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

set(SOLVER_SOURCES "${CMAKE_SOURCE_DIR}/src/solver.cpp")
set(BINDER_SOURCE  "${CMAKE_SOURCE_DIR}/src/binder.cpp")

add_library(solver_lib STATIC ${SOLVER_SOURCES})

target_include_directories(solver_lib PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(solver_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Eigen3::Eigen
    spdlog::spdlog
)

target_compile_options(solver_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)

pybind11_add_module(membrane MODULE ${BINDER_SOURCE})

target_include_directories(membrane PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(membrane PRIVATE
    solver_lib
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Eigen3::Eigen
    spdlog::spdlog
)

target_compile_options(membrane PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)

set_target_properties(membrane PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/run"
)

add_custom_command(TARGET membrane POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "membrane module built in run/"
)

install(TARGETS membrane
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/run"
)
