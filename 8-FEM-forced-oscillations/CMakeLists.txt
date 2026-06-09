cmake_minimum_required(VERSION 3.18)
project(fd_solver_pybind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

add_library(fd_solver_core STATIC
    src/FD_solver.cpp
)
target_include_directories(fd_solver_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(fd_solver_core PUBLIC cxx_std_17)

pybind11_add_module(fd_solver
    src/pybind_module.cpp
)

target_link_libraries(fd_solver PRIVATE fd_solver_core)

set_target_properties(fd_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/run
)
