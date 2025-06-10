#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// Sobel edge detection kernel
__global__ void sobel_edge_detection_kernel(uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Sobel operatörleri
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    int gx = 0, gy = 0;
    
    // 3x3 convolution
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            // Boundary check
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }
            
            uint8_t pixel = input[ny * width + nx];
            
            gx += pixel * sobel_x[dy + 1][dx + 1];
            gy += pixel * sobel_y[dy + 1][dx + 1];
        }
    }
    
    // Gradient magnitude
    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    
    // Clamp to 0-255
    magnitude = min(255, max(0, magnitude));
    
    output[y * width + x] = (uint8_t)magnitude;
}

// RGB to Grayscale conversion kernel
__global__ void rgb_to_grayscale_kernel(uint8_t* rgb_input, uint8_t* gray_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int rgb_idx = idx * 3;
    
    // RGB to Grayscale: 0.299*R + 0.587*G + 0.114*B
    uint8_t r = rgb_input[rgb_idx];
    uint8_t g = rgb_input[rgb_idx + 1];
    uint8_t b = rgb_input[rgb_idx + 2];
    
    uint8_t gray = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    gray_output[idx] = gray;
}

// Optimized Sobel with shared memory
__global__ void sobel_edge_detection_shared_kernel(uint8_t* input, uint8_t* output, int width, int height) {
    // Shared memory for tile + halo
    __shared__ uint8_t tile[18][18]; // 16x16 + 2 pixel halo
    
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    int global_x = block_x + thread_x;
    int global_y = block_y + thread_y;
    
    // Load tile with halo into shared memory
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int load_x = global_x + dx;
            int load_y = global_y + dy;
            int shared_x = thread_x + dx + 1;
            int shared_y = thread_y + dy + 1;
            
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
                tile[shared_y][shared_x] = input[load_y * width + load_x];
            } else {
                tile[shared_y][shared_x] = 0; // Zero padding
            }
        }
    }
    
    __syncthreads();
    
    if (global_x >= width || global_y >= height) return;
    
    // Sobel operators
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    int gx = 0, gy = 0;
    int center_x = thread_x + 1;
    int center_y = thread_y + 1;
    
    // Apply Sobel operators using shared memory
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            uint8_t pixel = tile[center_y + dy][center_x + dx];
            gx += pixel * sobel_x[dy + 1][dx + 1];
            gy += pixel * sobel_y[dy + 1][dx + 1];
        }
    }
    
    // Calculate magnitude
    int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
    magnitude = min(255, max(0, magnitude));
    
    output[global_y * width + global_x] = (uint8_t)magnitude;
}

// Launcher functions
extern "C" {
    void launch_sobel_kernel(uint8_t* input, uint8_t* output, int width, int height, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                       (height + block_size.y - 1) / block_size.y);
        
        // Use optimized shared memory version for better performance
        sobel_edge_detection_shared_kernel<<<grid_size, block_size, 0, stream>>>(input, output, width, height);
    }
    
    void launch_rgb_to_gray_kernel(uint8_t* rgb_input, uint8_t* gray_output, int width, int height, cudaStream_t stream) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                       (height + block_size.y - 1) / block_size.y);
        
        rgb_to_grayscale_kernel<<<grid_size, block_size, 0, stream>>>(rgb_input, gray_output, width, height);
    }
}