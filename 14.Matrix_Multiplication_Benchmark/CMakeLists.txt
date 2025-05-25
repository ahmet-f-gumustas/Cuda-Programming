cmake_minimum_required(VERSION 3.18)
project(MatrixMultiplicationBenchmark LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES 75 86 87 89)  # RTX 4070 destekli mimariler

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/matrix_operations.cu
    src/benchmark.cpp
    src/utils.cpp
)

# Create executable
add_executable(matrix_benchmark ${SOURCES})

# Set CUDA properties
set_property(TARGET matrix_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Compiler flags
target_compile_options(matrix_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -Xcompiler -Wall
        -Xcompiler -Wextra
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
        -Wextra
        -march=native
    >
)

# Link libraries
target_link_libraries(matrix_benchmark 
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
)

# Debug build için ek ayarlar
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(matrix_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Install target
install(TARGETS matrix_benchmark DESTINATION bin)