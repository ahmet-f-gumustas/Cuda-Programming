#ifndef CUDA_EDGE_DETECTOR_H
#define CUDA_EDGE_DETECTOR_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <memory>

class CudaEdgeDetector {
private:
    uint8_t *d_input_frame;
    uint8_t *d_output_frame;
    uint8_t *d_gray_frame;
    
    size_t input_size;
    size_t output_size;
    int frame_width;
    int frame_height;
    
    bool is_initialized;
    
    // CUDA streams for async processing
    cudaStream_t stream;
    
public:
    CudaEdgeDetector();
    ~CudaEdgeDetector();
    
    bool initialize();
    void cleanup();
    
    bool process_frame(const uint8_t* input_data, size_t data_size,
                      int width, int height,
                      uint8_t** output_data, size_t* output_size);
    
    // Sobel edge detection kernel wrapper
    bool apply_sobel_edge_detection(const uint8_t* input, uint8_t* output,
                                   int width, int height);
    
    // RGB to Grayscale conversion
    bool rgb_to_grayscale(const uint8_t* rgb_input, uint8_t* gray_output,
                         int width, int height);
    
    // Memory allocation helpers
    bool allocate_gpu_memory(int width, int height);
    void deallocate_gpu_memory();
};

// CUDA kernel declarations
extern "C" {
    void launch_sobel_kernel(uint8_t* input, uint8_t* output, int width, int height, cudaStream_t stream);
    void launch_rgb_to_gray_kernel(uint8_t* rgb_input, uint8_t* gray_output, int width, int height, cudaStream_t stream);
}

#endif // CUDA_EDGE_DETECTOR_H