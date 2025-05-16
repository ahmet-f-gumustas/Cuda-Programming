cmake_minimum_required(VERSION 3.10)
project(ParallelVision LANGUAGES CXX CUDA)

# C++17 standardı kullanıyoruz
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA bağımlılıkları
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV bağımlılığını ekliyoruz (görüntü işleme için)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include klasörleri
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA dosyalarını ayrı olarak belirtiyoruz
set(CUDA_SOURCES
    src/cuda_kernels.cu
    src/image_processor.cu
)

# CPP dosyalarını topluyoruz
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.cuh")

# Çalıştırılabilir dosya
cuda_add_executable(parallelvision ${CPP_SOURCES} ${CUDA_SOURCES} ${HEADERS})

# CUDA flags
set_target_properties(parallelvision PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Bağımlılıkları bağlıyoruz
target_link_libraries(parallelvision ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Kurulum hedefi
install(TARGETS parallelvision DESTINATION bin)
