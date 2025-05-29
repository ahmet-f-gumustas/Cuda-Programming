cmake_minimum_required(VERSION 3.18)
project(ParallelAlgorithmsCollection CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4070 Ti Super için Ada Lovelace

# CUDA ve gerekli paketleri bul
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas --allow-expensive-optimizations=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Debug ve Release ayarları
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

# Thrust kütüphanesini dahil et
thrust_create_target(Thrust)

# Kaynak dosyaları
set(ALGORITHM_SOURCES
    src/prefix_sum.cu
    src/reduce.cu
    src/histogram.cu
    src/radix_sort.cu
    src/bfs.cu
    src/scan.cu
    src/compact.cu
    src/matrix_multiply.cu
    src/merge_sort.cu
    src/convolution.cu
)

set(COMMON_SOURCES
    src/common/cuda_utils.cu
    src/common/timer.cpp
    src/common/memory_manager.cu
)

# Ana executable
add_executable(parallel_algorithms
    src/main.cu
    ${ALGORITHM_SOURCES}
    ${COMMON_SOURCES}
)

# Thrust ile link et
target_link_libraries(parallel_algorithms Thrust)

# Test executable
add_executable(run_tests
    tests/test_main.cu
    ${ALGORITHM_SOURCES}
    ${COMMON_SOURCES}
)

target_link_libraries(run_tests Thrust)

# Benchmark executable
add_executable(run_benchmarks
    benchmarks/benchmark_main.cu
    ${ALGORITHM_SOURCES}
    ${COMMON_SOURCES}
)

target_link_libraries(run_benchmarks Thrust)

# Custom targets
add_custom_target(profile
    COMMAND nvprof --print-gpu-trace --log-file profile.log $<TARGET_FILE:parallel_algorithms>
    DEPENDS parallel_algorithms
    COMMENT "CUDA profiling ile çalıştırılıyor..."
)

add_custom_target(occupancy
    COMMAND nvprof --metrics achieved_occupancy,sm_efficiency $<TARGET_FILE:parallel_algorithms>
    DEPENDS parallel_algorithms
    COMMENT "Occupancy analizi yapılıyor..."
)

# Derleme bilgileri
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Compiler: ${CMAKE_CUDA_COMPILER}")