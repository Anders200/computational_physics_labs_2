cmake_minimum_required(VERSION 3.18)
project(FEM_solver_pybind LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release flags ────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(pybind11  REQUIRED)
find_package(Eigen3    REQUIRED NO_MODULE)
find_package(spdlog    REQUIRED)
find_package(fmt       REQUIRED)

# ── Static core library ──────────────────────────────────────────────────────
add_library(FEM_solver_core STATIC
    src/FEM.cpp
)

target_include_directories(FEM_solver_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(FEM_solver_core PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
)

target_compile_features(FEM_solver_core PUBLIC cxx_std_17)

# Required when the static lib is linked into a shared object (pybind11 .so)
set_target_properties(FEM_solver_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── Python extension (pybind11) ──────────────────────────────────────────────
pybind11_add_module(FEM_solver
    src/binder.cpp
)

target_link_libraries(FEM_solver PRIVATE
    FEM_solver_core
)

# Output the .so directly into run/ so Python can import it there
set_target_properties(FEM_solver PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/run
    RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/run
    ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/run
)
