cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project configuration
project(gpu_video_pipeline 
    VERSION 1.0.0
    DESCRIPTION "CUDA/GPU Accelerated Video Encode-Decode Pipeline"
    LANGUAGES CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -DDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Platform detection
if(EXISTS "/sys/firmware/devicetree/base/model")
    file(READ "/sys/firmware/devicetree/base/model" DEVICE_MODEL)
    if(DEVICE_MODEL MATCHES "Jetson")
        set(IS_JETSON_PLATFORM TRUE)
        message(STATUS "Jetson platform detected: ${DEVICE_MODEL}")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# GStreamer
pkg_check_modules(GSTREAMER REQUIRED 
    gstreamer-1.0>=1.14
    gstreamer-app-1.0>=1.14
    gstreamer-video-1.0>=1.14
)

if(GSTREAMER_FOUND)
    message(STATUS "GStreamer found: ${GSTREAMER_VERSION}")
    include_directories(${GSTREAMER_INCLUDE_DIRS})
    link_directories(${GSTREAMER_LIBRARY_DIRS})
else()
    message(FATAL_ERROR "GStreamer not found!")
endif()

# CUDA architecture detection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(IS_JETSON_PLATFORM)
        # Jetson architectures
        set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87")
        message(STATUS "Jetson CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        # Desktop GPU architectures
        set(CMAKE_CUDA_ARCHITECTURES "52;61;75;86;89")
        message(STATUS "Desktop CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Source files
set(HEADER_FILES
    include/gpu_pipeline.h
    include/cpu_pipeline.h
    include/cuda_edge_detector.h
    include/performance_monitor.h
)

set(SOURCE_FILES
    src/main.cpp
    src/gpu_pipeline.cpp
    src/cpu_pipeline.cpp
    src/cuda_edge_detector.cpp
    src/performance_monitor.cpp
)

set(CUDA_SOURCE_FILES
    src/cuda_kernels.cu
)

# Create executable
add_executable(${PROJECT_NAME}
    ${SOURCE_FILES}
    ${CUDA_SOURCE_FILES}
    ${HEADER_FILES}
)

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    -DWITH_CUDA
    -DPROJECT_VERSION="${PROJECT_VERSION}"
)

if(IS_JETSON_PLATFORM)
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DIS_JETSON_PLATFORM)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    Threads::Threads
)

# CUDA-specific compiler options
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# Custom targets
add_custom_target(test_video
    COMMAND gst-launch-1.0 videotestsrc num-buffers=1800 pattern=smpte
        ! video/x-raw,width=1920,height=1080,framerate=60/1
        ! x264enc bitrate=8000 speed-preset=ultrafast
        ! h264parse ! filesink location=test_input_1080p60.h264
    COMMENT "Creating test video file"
)

add_custom_target(run_pipeline
    COMMAND $<TARGET_FILE:${PROJECT_NAME}>
    DEPENDS ${PROJECT_NAME}
    COMMENT "Running the GPU pipeline"
)

add_custom_target(generate_report
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/generate_graphs.py performance_stats.csv --report
    COMMENT "Generating performance report"
)

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(FILES ${CMAKE_SOURCE_DIR}/scripts/generate_graphs.py
    DESTINATION bin
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ
)

# Print build information
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Jetson Platform: ${IS_JETSON_PLATFORM}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")