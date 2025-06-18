cmake_minimum_required(VERSION 3.18)
project(cuda_histogram LANGUAGES CXX CUDA)

# CUDA ayarları
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDA mimarisi ayarları (RTX 4070 için)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Include dizinleri
include_directories(include)

# Kaynak dosyaları
set(SOURCES
    src/main.cpp
    src/histogram_cpu.cpp
    src/histogram_cuda.cu
    src/utils.cpp
)

# Headers
set(HEADERS
    include/histogram_cpu.h
    include/histogram_cuda.h
    include/utils.h
)

# Executable oluştur
add_executable(cuda_histogram ${SOURCES} ${HEADERS})

# CUDA özelliklerini etkinleştir
set_property(TARGET cuda_histogram PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Compiler bayrakları
target_compile_options(cuda_histogram PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
)

# CUDA kütüphanelerini bağla
target_link_libraries(cuda_histogram ${CUDA_LIBRARIES})

# Debug ve Release konfigürasyonları
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(cuda_histogram PRIVATE DEBUG=1)
    target_compile_options(cuda_histogram PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Install kuralları
install(TARGETS cuda_histogram DESTINATION bin)