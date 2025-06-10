#include "cuda_edge_detector.h"
#include <iostream>
#include <cstring>

CudaEdgeDetector::CudaEdgeDetector() 
    : d_input_frame(nullptr), d_output_frame(nullptr), d_gray_frame(nullptr),
      input_size(0), output_size(0), frame_width(0), frame_height(0),
      is_initialized(false), stream(0) {
}

CudaEdgeDetector::~CudaEdgeDetector() {
    cleanup();
}

bool CudaEdgeDetector::initialize() {
    std::cout << "Initializing CUDA Edge Detector..." << std::endl;
    
    // CUDA stream oluştur
    cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    is_initialized = true;
    std::cout << "CUDA Edge Detector initialized successfully" << std::endl;
    return true;
}

void CudaEdgeDetector::cleanup() {
    deallocate_gpu_memory();
    
    if (stream) {
        cudaStreamDestroy(stream);
        stream = 0;
    }
    
    is_initialized = false;
}

bool CudaEdgeDetector::allocate_gpu_memory(int width, int height) {
    if (frame_width == width && frame_height == height && d_input_frame) {
        // Already allocated for this size
        return true;
    }
    
    // Clean up previous allocation
    deallocate_gpu_memory();
    
    frame_width = width;
    frame_height = height;
    
    // I420 format: Y plane + U/V planes
    input_size = width * height * 3 / 2; // I420 size
    output_size = input_size; // Same output size
    
    // Allocate GPU memory
    cudaError_t status = cudaMalloc(&d_input_frame, input_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate input GPU memory: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    status = cudaMalloc(&d_output_frame, output_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate output GPU memory: " << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input_frame);
        d_input_frame = nullptr;
        return false;
    }
    
    // Grayscale frame for edge detection (Y plane only)
    size_t gray_size = width * height;
    status = cudaMalloc(&d_gray_frame, gray_size);
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate grayscale GPU memory: " << cudaGetErrorString(status) << std::endl;
        cudaFree(d_input_frame);
        cudaFree(d_output_frame);
        d_input_frame = nullptr;
        d_output_frame = nullptr;
        return false;
    }
    
    std::cout << "GPU memory allocated: " << input_size << " bytes" << std::endl;
    return true;
}

void CudaEdgeDetector::deallocate_gpu_memory() {
    if (d_input_frame) {
        cudaFree(d_input_frame);
        d_input_frame = nullptr;
    }
    
    if (d_output_frame) {
        cudaFree(d_output_frame);
        d_output_frame = nullptr;
    }
    
    if (d_gray_frame) {
        cudaFree(d_gray_frame);
        d_gray_frame = nullptr;
    }
    
    frame_width = 0;
    frame_height = 0;
    input_size = 0;
    output_size = 0;
}

bool CudaEdgeDetector::process_frame(const uint8_t* input_data, size_t data_size,
                                   int width, int height,
                                   uint8_t** output_data, size_t* output_size_ptr) {
    if (!is_initialized) {
        std::cerr << "CUDA Edge Detector not initialized" << std::endl;
        return false;
    }
    
    // Memory allocate et (gerekirse)
    if (!allocate_gpu_memory(width, height)) {
        return false;
    }
    
    // Input data'yı GPU'ya kopyala
    cudaError_t status = cudaMemcpyAsync(d_input_frame, input_data, data_size, 
                                        cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy input data to GPU: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // I420 format'ta Y plane zaten grayscale, sadece edge detection uygula
    // Y plane = first width*height bytes
    if (!apply_sobel_edge_detection(d_input_frame, d_gray_frame, width, height)) {
        std::cerr << "Failed to apply edge detection" << std::endl;
        return false;
    }
    
    // Output frame'i hazırla: Edge detected Y plane + original U/V planes
    // Y plane'i edge detected version ile değiştir
    status = cudaMemcpyAsync(d_output_frame, d_gray_frame, width * height, 
                            cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy Y plane: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // U/V planes'i orijinal halinde kopyala
    size_t uv_offset = width * height;
    size_t uv_size = width * height / 2;
    status = cudaMemcpyAsync(d_output_frame + uv_offset, d_input_frame + uv_offset, uv_size,
                            cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy UV planes: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    
    // Host memory allocate et
    uint8_t* host_output = (uint8_t*)malloc(output_size);
    if (!host_output) {
        std::cerr << "Failed to allocate host output memory" << std::endl;
        return false;
    }
    
    // GPU'dan host'a kopyala
    status = cudaMemcpyAsync(host_output, d_output_frame, output_size,
                            cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy output data to host: " << cudaGetErrorString(status) << std::endl;
        free(host_output);
        return false;
    }
    
    // Stream'i bekle
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        std::cerr << "Failed to synchronize CUDA stream: " << cudaGetErrorString(status) << std::endl;
        free(host_output);
        return false;
    }
    
    *output_data = host_output;
    *output_size_ptr = output_size;
    
    return true;
}

bool CudaEdgeDetector::apply_sobel_edge_detection(const uint8_t* input, uint8_t* output,
                                                 int width, int height) {
    try {
        launch_sobel_kernel(const_cast<uint8_t*>(input), output, width, height, stream);
        
        // Kernel hatasını kontrol et
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::cerr << "Sobel kernel failed: " << cudaGetErrorString(status) << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in apply_sobel_edge_detection: " << e.what() << std::endl;
        return false;
    }
}

bool CudaEdgeDetector::rgb_to_grayscale(const uint8_t* rgb_input, uint8_t* gray_output,
                                       int width, int height) {
    try {
        launch_rgb_to_gray_kernel(const_cast<uint8_t*>(rgb_input), gray_output, width, height, stream);
        
        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::cerr << "RGB to Gray kernel failed: " << cudaGetErrorString(status) << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in rgb_to_grayscale: " << e.what() << std::endl;
        return false;
    }
}