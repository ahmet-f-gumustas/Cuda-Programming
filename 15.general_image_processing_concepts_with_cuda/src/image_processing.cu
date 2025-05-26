#include "image_processing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

// Constants
#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 15

// Gaussian kernel constant memory
__constant__ float d_gaussianKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// Sobel kernels
__constant__ float d_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ float d_sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// ===== CUDA KERNELS =====

// Grayscale conversion kernel
__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, 
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * 3; // RGB input
        int output_idx = y * width + x;       // Grayscale output
        
        // Grayscale formula: 0.299*R + 0.587*G + 0.114*B
        float gray = 0.299f * input[pixel_idx] + 
                     0.587f * input[pixel_idx + 1] + 
                     0.114f * input[pixel_idx + 2];
        
        output[output_idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, gray));
    }
}

// Gaussian blur kernel with shared memory
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output,
                                  int width, int height, int channels, int kernelSize) {
    extern __shared__ unsigned char sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    int sharedWidth = blockDim.x + kernelSize - 1;
    int sharedHeight = blockDim.y + kernelSize - 1;
    int halfKernel = kernelSize / 2;
    
    // Load data to shared memory
    for (int c = 0; c < channels; c++) {
        int sharedIdx = (ty + halfKernel) * sharedWidth + (tx + halfKernel);
        sharedIdx = sharedIdx * channels + c;
        
        if (x < width && y < height) {
            int globalIdx = (y * width + x) * channels + c;
            sharedMem[sharedIdx] = input[globalIdx];
        } else {
            sharedMem[sharedIdx] = 0;
        }
    }
    
    // Load halo regions
    // Top and bottom
    if (ty < halfKernel) {
        for (int c = 0; c < channels; c++) {
            // Top
            int topY = y - halfKernel;
            if (topY >= 0 && x < width) {
                int globalIdx = (topY * width + x) * channels + c;
                int sharedIdx = (ty * sharedWidth + tx + halfKernel) * channels + c;
                sharedMem[sharedIdx] = input[globalIdx];
            }
            
            // Bottom
            int bottomY = y + blockDim.y;
            if (bottomY < height && x < width) {
                int globalIdx = (bottomY * width + x) * channels + c;
                int sharedIdx = ((ty + blockDim.y + halfKernel) * sharedWidth + tx + halfKernel) * channels + c;
                sharedMem[sharedIdx] = input[globalIdx];
            }
        }
    }
    
    // Left and right
    if (tx < halfKernel) {
        for (int c = 0; c < channels; c++) {
            // Left
            int leftX = x - halfKernel;
            if (leftX >= 0 && y < height) {
                int globalIdx = (y * width + leftX) * channels + c;
                int sharedIdx = ((ty + halfKernel) * sharedWidth + tx) * channels + c;
                sharedMem[sharedIdx] = input[globalIdx];
            }
            
            // Right
            int rightX = x + blockDim.x;
            if (rightX < width && y < height) {
                int globalIdx = (y * width + rightX) * channels + c;
                int sharedIdx = ((ty + halfKernel) * sharedWidth + tx + blockDim.x + halfKernel) * channels + c;
                sharedMem[sharedIdx] = input[globalIdx];
            }
        }
    }
    
    __syncthreads();
    
    // Apply Gaussian filter
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernelSize; ky++) {
                for (int kx = 0; kx < kernelSize; kx++) {
                    int sharedIdx = ((ty + ky) * sharedWidth + (tx + kx)) * channels + c;
                    float kernelValue = d_gaussianKernel[ky * kernelSize + kx];
                    sum += kernelValue * sharedMem[sharedIdx];
                }
            }
            
            int outputIdx = (y * width + x) * channels + c;
            output[outputIdx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum));
        }
    }
}

// Sobel edge detection kernel
__global__ void sobelKernel(unsigned char* input, unsigned char* output,
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float gx = 0.0f, gy = 0.0f;
        
        // Apply Sobel operators
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixel_idx = ((y + ky) * width + (x + kx)) * 3;
                
                // Convert to grayscale first
                float gray = 0.299f * input[pixel_idx] + 
                            0.587f * input[pixel_idx + 1] + 
                            0.114f * input[pixel_idx + 2];
                
                int kernel_idx = (ky + 1) * 3 + (kx + 1);
                gx += d_sobelX[kernel_idx] * gray;
                gy += d_sobelY[kernel_idx] * gray;
            }
        }
        
        // Calculate gradient magnitude
        float magnitude = sqrtf(gx * gx + gy * gy);
        int output_idx = y * width + x;
        output[output_idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, magnitude));
    }
}

// Brightness adjustment kernel
__global__ void brightnessKernel(unsigned char* input, unsigned char* output,
                                int width, int height, int channels, float factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            float newValue = input[idx] * factor;
            output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, newValue));
        }
    }
}

// ===== HOST FUNCTIONS =====

// Generate Gaussian kernel
void generateGaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int half = size / 2;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

// ===== KERNEL LAUNCH FUNCTIONS =====

extern "C" void launchGrayscaleKernel(unsigned char* d_input, unsigned char* d_output,
                                     int width, int height) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void launchGaussianBlurKernel(unsigned char* d_input, unsigned char* d_output,
                                        int width, int height, float sigma) {
    // Generate and copy Gaussian kernel
    int kernelSize = (int)(6 * sigma) | 1; // Ensure odd size
    if (kernelSize > MAX_KERNEL_SIZE) kernelSize = MAX_KERNEL_SIZE;
    
    float h_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
    generateGaussianKernel(h_kernel, kernelSize, sigma);
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_gaussianKernel, h_kernel, 
                                  kernelSize * kernelSize * sizeof(float)));
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Calculate shared memory size
    int sharedWidth = blockSize.x + kernelSize - 1;
    int sharedHeight = blockSize.y + kernelSize - 1;
    size_t sharedMemSize = sharedWidth * sharedHeight * 3 * sizeof(unsigned char);
    
    gaussianBlurKernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_input, d_output, width, height, 3, kernelSize);
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void launchSobelKernel(unsigned char* d_input, unsigned char* d_output,
                                 int width, int height) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void launchBrightnessKernel(unsigned char* d_input, unsigned char* d_output,
                                      int width, int height, float factor) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    brightnessKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, 3, factor);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ===== ImageProcessor CLASS IMPLEMENTATION =====

ImageProcessor::ImageProcessor() : d_input(nullptr), d_output(nullptr), 
                                   input_size(0), output_size(0) {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
}

ImageProcessor::~ImageProcessor() {
    freeMemory();
}

void ImageProcessor::allocateMemory(int width, int height, int input_channels, int output_channels) {
    freeMemory(); // Free any existing memory
    
    input_size = width * height * input_channels * sizeof(unsigned char);
    output_size = width * height * output_channels * sizeof(unsigned char);
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
}

void ImageProcessor::freeMemory() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

void ImageProcessor::processGrayscale(const Image& input, Image& output) {
    allocateMemory(input.width, input.height, input.channels, 1);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launchGrayscaleKernel(d_input, d_output, input.width, input.height);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));
}

void ImageProcessor::processGaussianBlur(const Image& input, Image& output, float sigma) {
    allocateMemory(input.width, input.height, input.channels, input.channels);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launchGaussianBlurKernel(d_input, d_output, input.width, input.height, sigma);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));
}

void ImageProcessor::processSobel(const Image& input, Image& output) {
    allocateMemory(input.width, input.height, input.channels, 1);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launchSobelKernel(d_input, d_output, input.width, input.height);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));
}

void ImageProcessor::processBrightness(const Image& input, Image& output, float factor) {
    allocateMemory(input.width, input.height, input.channels, input.channels);
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input.data, input_size, cudaMemcpyHostToDevice));
    
    // Launch kernel
    launchBrightnessKernel(d_input, d_output, input.width, input.height, factor);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data, d_output, output_size, cudaMemcpyDeviceToHost));
}

void ImageProcessor::printDeviceInfo() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "CUDA Devices found: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\nDevice " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    }
    
    // Set device 0 as active
    CUDA_CHECK(cudaSetDevice(0));
}