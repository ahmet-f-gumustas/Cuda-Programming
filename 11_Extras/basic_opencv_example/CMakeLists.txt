cmake_minimum_required(VERSION 3.10)
project(opencv_cuda_example)

# C++11 standardını kullan
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA bul
find_package(CUDA REQUIRED)
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")

# OpenCV bul
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaarithm cudafilters)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

# CUDA mimari ve kod oluşturma seçenekleri için
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; 
    -O3
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_52,code=sm_52
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_70,code=sm_70
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_80,code=sm_80
    -gencode arch=compute_86,code=sm_86
)

# Include dizinlerini ekle
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Derlenecek kaynak dosyalarını belirt
add_executable(opencv_cuda_app main.cpp)

# Hedefi bağla
target_link_libraries(opencv_cuda_app ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# Derleme tamamlandıktan sonra bilgi mesajı
message(STATUS "Configuration complete - use 'make' to build the project")