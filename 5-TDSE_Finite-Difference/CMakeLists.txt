cmake_minimum_required(VERSION 3.18)
project(TdSE_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ── Python discovery ──────────────────────────────────────────────────────────
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
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

set(PYBIND11_FINDPYTHON ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT DEFINED PYBIND11_PYTHON_EXECUTABLE)
    set(PYBIND11_PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
        CACHE PATH "Python interpreter for pybind11" FORCE)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(pybind11  REQUIRED CONFIG)
find_package(spdlog    REQUIRED)
find_package(fmt       REQUIRED)

find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")

# ── itm Module ────────────────────────────────────────────────────────────────────
pybind11_add_module(
    itm
    src/itm.cpp
    src/TdSE_binder.cpp
    src/itm_binder.cpp
    src/TdSE.cpp
)

target_include_directories(itm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(itm PRIVATE
    pybind11::module
    Python3::Module
    spdlog::spdlog
    fmt::fmt
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver    
    CUDA::cusparse    
)

# ── TdSE Module ────────────────────────────────────────────────────────────────────
pybind11_add_module(
    TdSE 
    src/TdSE.cpp
    src/TdSE_binder.cpp
)

# Fixed: Changed target from 'itm' to 'TdSE'
target_include_directories(TdSE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Fixed: Changed target from 'itm' to 'TdSE'
target_link_libraries(TdSE PRIVATE
    pybind11::module
    Python3::Module
    spdlog::spdlog
    fmt::fmt
)

# ── Output location ───────────────────────────────────────────────────────────
set_target_properties(itm TdSE PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/run"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/run"
)
