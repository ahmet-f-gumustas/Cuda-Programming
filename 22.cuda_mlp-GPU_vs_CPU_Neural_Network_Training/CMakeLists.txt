# CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(cuda_mlp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Main executable
add_executable(cuda_mlp
    src/main.cpp
    src/neural_net_cpu.cpp
    src/neural_net_gpu.cu
    src/utils.cpp
)

target_link_libraries(cuda_mlp
    CUDA::cublas
    CUDA::cudart
)

# Test executable
add_executable(mnist_accuracy
    tests/mnist_accuracy.cpp
    src/neural_net_cpu.cpp
    src/neural_net_gpu.cu
    src/utils.cpp
)

target_link_libraries(mnist_accuracy
    CUDA::cublas
    CUDA::cudart
)

# Enable CUDA separable compilation
set_target_properties(cuda_mlp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mnist_accuracy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)