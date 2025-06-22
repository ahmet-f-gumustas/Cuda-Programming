cmake_minimum_required(VERSION 3.18)

# Project name and version
project(CudaVectorAddition VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Set CUDA architecture for RTX 4070 (Ada Lovelace - sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Print some useful information
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add executable
add_executable(vector_add src/vector_add.cu)

# Set target properties
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries
target_link_libraries(vector_add ${CUDA_LIBRARIES})

# Optional: Add more advanced version with timing and verification
add_executable(vector_add_advanced src/vector_add_advanced.cu)
set_target_properties(vector_add_advanced PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(vector_add_advanced ${CUDA_LIBRARIES})

# Custom target for running the program
add_custom_target(run
    COMMAND vector_add
    DEPENDS vector_add
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
    COMMENT "Running CUDA Vector Addition"
)

add_custom_target(run_advanced
    COMMAND vector_add_advanced
    DEPENDS vector_add_advanced
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
    COMMENT "Running Advanced CUDA Vector Addition"
)

# Custom target for checking CUDA installation
add_custom_target(cuda_info
    COMMAND nvidia-smi
    COMMAND nvcc --version
    COMMENT "Displaying CUDA information"
)

# Install targets
install(TARGETS vector_add vector_add_advanced
    RUNTIME DESTINATION bin
)

# Print final message
message(STATUS "Configuration completed successfully!")
message(STATUS "To build: mkdir build && cd build && cmake .. && make")
message(STATUS "To run: make run or make run_advanced")