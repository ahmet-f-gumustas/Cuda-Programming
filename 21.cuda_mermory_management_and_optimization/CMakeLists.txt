cmake_minimum_required(VERSION 3.18)
project(cuda_memory_management CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA ayarları
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include dizinleri
include_directories(include)

# CUDA derleme bayrakları
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_89") # RTX 4070 için
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Kaynak dosyaları
set(CUDA_SOURCES
    src/memory_manager.cu
    src/global_memory_example.cu
    src/shared_memory_example.cu
    src/constant_memory_example.cu
    src/unified_memory_example.cu
)

# Executable
add_executable(memory_benchmark src/main.cpp ${CUDA_SOURCES})

# CUDA kütüphanelerini linkle
target_link_libraries(memory_benchmark ${CUDA_LIBRARIES})

# Optimizasyon seviyeleri
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()