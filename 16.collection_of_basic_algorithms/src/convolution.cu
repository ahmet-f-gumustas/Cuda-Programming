// ==================== src/convolution.cu ====================
#include "../include/common.h"

// 1D convolution kernel
__global__ void convolution_1d_kernel(const float* signal, const float* kernel, 
                                      float* output, int signal_size, int kernel_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < signal_size) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int k = 0; k < kernel_size; ++k) {
            int signal_idx = tid - half_kernel + k;
            if (signal_idx >= 0 && signal_idx < signal_size) {
                sum += signal[signal_idx] * kernel[k];
            }
        }
        
        output[tid] = sum;
    }
}

// 1D convolution with shared memory
__global__ void convolution_1d_shared_kernel(const float* signal, const float* kernel,
                                            float* output, int signal_size, int kernel_size) {
    extern __shared__ float shared_signal[];
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int half_kernel = kernel_size / 2;
    
    // Load signal data into shared memory with halo
    int shared_size = blockDim.x + kernel_size - 1;
    int start_idx = blockIdx.x * blockDim.x - half_kernel;
    
    // Load main data
    if (start_idx + tid >= 0 && start_idx + tid < signal_size) {
        shared_signal[tid] = signal[start_idx + tid];
    } else {
        shared_signal[tid] = 0.0f; // Zero padding
    }
    
    // Load halo data
    if (tid < kernel_size - 1) {
        int halo_idx = start_idx + blockDim.x + tid;
        if (halo_idx < signal_size) {
            shared_signal[blockDim.x + tid] = signal[halo_idx];
        } else {
            shared_signal[blockDim.x + tid] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (global_id < signal_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; ++k) {
            sum += shared_signal[tid + k] * kernel[k];
        }
        output[global_id] = sum;
    }
}

// 2D convolution kernel
__global__ void convolution_2d_kernel(const float* image, const float* kernel,
                                     float* output, int width, int height, int kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int image_row = row - half_kernel + ky;
                int image_col = col - half_kernel + kx;
                
                if (image_row >= 0 && image_row < height && 
                    image_col >= 0 && image_col < width) {
                    int image_idx = image_row * width + image_col;
                    int kernel_idx = ky * kernel_size + kx;
                    sum += image[image_idx] * kernel[kernel_idx];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// 1D convolution implementation
void convolution_1d_custom(const std::vector<float>& signal, const std::vector<float>& kernel,
                          std::vector<float>& output) {
    int signal_size = signal.size();
    int kernel_size = kernel.size();
    output.resize(signal_size);
    
    ManagedMemory<float> d_signal(signal_size);
    ManagedMemory<float> d_kernel(kernel_size);
    ManagedMemory<float> d_output(signal_size);
    
    d_signal.copy_from_host(signal.data());
    d_kernel.copy_from_host(kernel.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (signal_size + threads_per_block - 1) / threads_per_block;
    
    convolution_1d_kernel<<<blocks, threads_per_block>>>(
        d_signal.get(), d_kernel.get(), d_output.get(), signal_size, kernel_size);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    // Calculate throughput
    double operations = (double)signal_size * kernel_size;
    double throughput = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "1D Convolution (Naive) - Time: " << timer.elapsed_ms() 
              << " ms, Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " GOPS" << std::endl;
}

// 1D convolution with shared memory
void convolution_1d_shared(const std::vector<float>& signal, const std::vector<float>& kernel,
                          std::vector<float>& output) {
    int signal_size = signal.size();
    int kernel_size = kernel.size();
    output.resize(signal_size);
    
    ManagedMemory<float> d_signal(signal_size);
    ManagedMemory<float> d_kernel(kernel_size);
    ManagedMemory<float> d_output(signal_size);
    
    d_signal.copy_from_host(signal.data());
    d_kernel.copy_from_host(kernel.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (signal_size + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = (threads_per_block + kernel_size - 1) * sizeof(float);
    
    convolution_1d_shared_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_signal.get(), d_kernel.get(), d_output.get(), signal_size, kernel_size);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    // Calculate throughput
    double operations = (double)signal_size * kernel_size;
    double throughput = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "1D Convolution (Shared) - Time: " << timer.elapsed_ms() 
              << " ms, Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " GOPS" << std::endl;
}

// 2D convolution implementation
void convolution_2d_custom(const std::vector<float>& image, const std::vector<float>& kernel,
                          std::vector<float>& output, int width, int height, int kernel_size) {
    output.resize(width * height);
    
    ManagedMemory<float> d_image(width * height);
    ManagedMemory<float> d_kernel(kernel_size * kernel_size);
    ManagedMemory<float> d_output(width * height);
    
    d_image.copy_from_host(image.data());
    d_kernel.copy_from_host(kernel.data());
    
    CudaTimer timer;
    timer.start();
    
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    convolution_2d_kernel<<<grid, block>>>(
        d_image.get(), d_kernel.get(), d_output.get(), width, height, kernel_size);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    // Calculate throughput
    double operations = (double)width * height * kernel_size * kernel_size;
    double throughput = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "2D Convolution - Time: " << timer.elapsed_ms() 
              << " ms, Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " GOPS" << std::endl;
}

// Test function
void test_convolution() {
    std::cout << "\n=== CONVOLUTION TEST ===" << std::endl;
    
    // 1D Convolution test
    const int signal_size = 1000000;
    const int kernel_size = 128;
    
    auto signal = generate_random_data<float>(signal_size, -1.0f, 1.0f);
    auto kernel_1d = generate_random_data<float>(kernel_size, -0.1f, 0.1f);
    
    std::cout << "1D Convolution test (signal: " << signal_size 
              << ", kernel: " << kernel_size << "):" << std::endl;
    
    std::vector<float> output_1d_naive, output_1d_shared;
    convolution_1d_custom(signal, kernel_1d, output_1d_naive);
    convolution_1d_shared(signal, kernel_1d, output_1d_shared);
    
    // Verify 1D results match
    bool results_1d_match = true;
    const float tolerance = 1e-5f;
    for (int i = 0; i < signal_size && results_1d_match; ++i) {
        if (std::abs(output_1d_naive[i] - output_1d_shared[i]) > tolerance) {
            results_1d_match = false;
        }
    }
    std::cout << "1D Convolution results match: " << (results_1d_match ? "✓" : "✗") << std::endl;
    
    // 2D Convolution test
    const int width = 1024;
    const int height = 1024;
    const int kernel_2d_size = 5;
    
    auto image = generate_random_data<float>(width * height, 0.0f, 255.0f);
    auto kernel_2d = generate_random_data<float>(kernel_2d_size * kernel_2d_size, -0.1f, 0.1f);
    
    std::cout << "\n2D Convolution test (image: " << width << "x" << height 
              << ", kernel: " << kernel_2d_size << "x" << kernel_2d_size << "):" << std::endl;
    
    std::vector<float> output_2d;
    convolution_2d_custom(image, kernel_2d, output_2d, width, height, kernel_2d_size);
    
    // Show sample results
    std::cout << "\nSample results (first 10 elements):" << std::endl;
    std::cout << "1D Signal: ";
    for (int i = 0; i < 10; ++i) std::cout << std::fixed << std::setprecision(3) << signal[i] << " ";
    std::cout << std::endl;
    
    std::cout << "1D Output: ";
    for (int i = 0; i < 10; ++i) std::cout << std::fixed << std::setprecision(3) << output_1d_naive[i] << " ";
    std::cout << std::endl;
    
    std::cout << "\nConvolution demonstrates the importance of memory access patterns" << std::endl;
    std::cout << "and shared memory optimization for stencil computations." << std::endl;
}