cmake_minimum_required(VERSION 3.18)
project(GStreamerCudaPointCloud LANGUAGES CXX CUDA)

# C++ standardı
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA ayarları
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89) # RTX 4070 için

# Debug/Release ayarları
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Paketleri bul
find_package(CUDA REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# GStreamer paketleri
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER_APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER_VIDEO REQUIRED gstreamer-video-1.0)

# Include dizinleri
include_directories(${GSTREAMER_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Executable oluştur
add_executable(pointcloud_processor main.cu)

# CUDA özelliklerini ayarla
set_property(TARGET pointcloud_processor PROPERTY CUDA_ARCHITECTURES 75 86 89)

# Bağlantı kütüphaneleri
target_link_libraries(pointcloud_processor
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_APP_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_RUNTIME_LIBRARY}
    Threads::Threads
)

# Compiler bayrakları
target_compile_options(pointcloud_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -O3
        --use_fast_math
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -Wall
        -Wextra
    >
)

# GStreamer bayrakları
target_compile_options(pointcloud_processor PRIVATE ${GSTREAMER_CFLAGS_OTHER})
target_link_directories(pointcloud_processor PRIVATE ${GSTREAMER_LIBRARY_DIRS})

# Debug bilgileri
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pointcloud_processor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    )
endif()

# Install kuralları
install(TARGETS pointcloud_processor
    RUNTIME DESTINATION bin
)