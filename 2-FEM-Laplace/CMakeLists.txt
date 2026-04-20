cmake_minimum_required(VERSION 3.14)
project(FEM_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

if(NOT DEFINED Python3_EXECUTABLE)
    if(DEFINED ENV{PYTHON3})
        set(Python3_EXECUTABLE $ENV{PYTHON3})
    elseif(DEFINED ENV{PYTHON})
        set(Python3_EXECUTABLE $ENV{PYTHON})
    else()
        execute_process(COMMAND which python3
                        OUTPUT_VARIABLE _PY_EXEC
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_PY_EXEC)
            set(Python3_EXECUTABLE ${_PY_EXEC})
        endif()
    endif()
endif()

message(STATUS "Requested Python3 executable: ${Python3_EXECUTABLE}")

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Make sure pybind11 will use the same Python interpreter
if(NOT DEFINED PYBIND11_PYTHON_EXECUTABLE)
    set(PYBIND11_PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE PATH "Python interpreter for pybind11" FORCE)
endif()

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(FEM
    src/FEM.cpp
    src/binder.cpp
)

target_include_directories(FEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(FEM PRIVATE
    pybind11::module
    Python3::Module
    spdlog::spdlog
    fmt::fmt
    CUDA::cublas
    CUDA::cudart
    CUDA::cusolver
)
set_target_properties(FEM PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/run"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/run"
)
