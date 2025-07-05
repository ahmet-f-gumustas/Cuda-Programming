#include "memory_manager.cuh"

__global__ void globalMemoryKernel(float* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Basit bir işlem: veriyi oku, işle ve geri yaz
        float value = d_data[idx];
        value = sqrtf(value * value + 1.0f);
        d_data[idx] = value;
    }
}

// Coalesced memory access örneği
__global__ void coalescedAccessKernel(float* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < size; i += stride) {
        d_data[i] = sqrtf(d_data[i] * 2.0f + 1.0f);
    }
}

// Non-coalesced memory access örneği (kötü performans)
__global__ void nonCoalescedAccessKernel(float* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size/32) {
        // Her thread 32 element atlar - kötü memory pattern
        for (int i = 0; i < 32; i++) {
            int index = idx + i * (size/32);
            if (index < size) {
                d_data[index] = sqrtf(d_data[index] * 2.0f + 1.0f);
            }
        }
    }
}

void runGlobalMemoryExample(int size) {
    float* h_data = new float[size];
    float* d_data;
    
    // Veriyi başlat
    for (int i = 0; i < size; i++) {
        h_data[i] = static_cast<float>(i);
    }
    
    // GPU belleği ayır
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Kernel konfigürasyonu
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Coalesced access
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    coalescedAccessKernel<<<gridSize, blockSize>>>(d_data, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float coalescedTime;
    CUDA_CHECK(cudaEventElapsedTime(&coalescedTime, start, stop));
    
    // Non-coalesced access
    CUDA_CHECK(cudaEventRecord(start));
    nonCoalescedAccessKernel<<<gridSize, blockSize>>>(d_data, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float nonCoalescedTime;
    CUDA_CHECK(cudaEventElapsedTime(&nonCoalescedTime, start, stop));
    
    std::cout << "  Coalesced access time: " << coalescedTime << " ms" << std::endl;
    std::cout << "  Non-coalesced access time: " << nonCoalescedTime << " ms" << std::endl;
    std::cout << "  Speedup: " << nonCoalescedTime / coalescedTime << "x" << std::endl;
    
    // Belleği temizle
    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}