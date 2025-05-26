cmake_minimum_required(VERSION 3.18)
project(ImageProcessingCUDA LANGUAGES CXX CUDA)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/image_processing.cu
    src/image_utils.cpp
)

# Create executable
add_executable(image_processor ${SOURCES})

# Set CUDA properties
set_target_properties(image_processor PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "75;86;89"  # RTX 4070 için
)

# Compiler flags
target_compile_options(image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Link libraries
target_link_libraries(image_processor ${CUDA_LIBRARIES})

# Debug build için ek flagler
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(image_processor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -lineinfo>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()