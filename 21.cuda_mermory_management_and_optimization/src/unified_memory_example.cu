#include "memory_manager.cuh"

// Unified Memory kullanan vector addition
__global__ void vectorAddUnified(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Prefetch ile optimize edilmiş versiyon
void runUnifiedMemoryWithPrefetch(int size) {
    float *a, *b, *c;
    
    // Unified Memory ayır
    CUDA_CHECK(cudaMallocManaged(&a, size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, size * sizeof(float)));
    
    // CPU'da başlat
    for (int i = 0; i < size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    // GPU device ID
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    
    // Prefetch to GPU
    CUDA_CHECK(cudaMemPrefetchAsync(a, size * sizeof(float), device));
    CUDA_CHECK(cudaMemPrefetchAsync(b, size * sizeof(float), device));
    CUDA_CHECK(cudaMemPrefetchAsync(c, size * sizeof(float), device));
    
    // Kernel çalıştır
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    vectorAddUnified<<<gridSize, blockSize>>>(a, b, c, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernelTime;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    
    std::cout << "    With prefetch: " << kernelTime << " ms" << std::endl;
    
    // Belleği temizle
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// Prefetch olmadan
void runUnifiedMemoryWithoutPrefetch(int size) {
    float *a, *b, *c;
    
    // Unified Memory ayır
    CUDA_CHECK(cudaMallocManaged(&a, size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&b, size * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&c, size * sizeof(float)));
    
    // CPU'da başlat
    for (int i = 0; i < size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    // Kernel çalıştır (prefetch yok)
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    vectorAddUnified<<<gridSize, blockSize>>>(a, b, c, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernelTime;
    CUDA_CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    
    std::cout << "    Without prefetch: " << kernelTime << " ms" << std::endl;
    
    // Belleği temizle
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void runUnifiedMemoryExample(int size) {
    std::cout << "  Unified Memory Performance:" << std::endl;
    
    // Prefetch olmadan
    runUnifiedMemoryWithoutPrefetch(size);
    
    // Prefetch ile
    runUnifiedMemoryWithPrefetch(size);
    
    // Memory advice örneği
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, size * sizeof(float)));
    
    // Memory advice set et
    int device = 0;
    CUDA_CHECK(cudaMemAdvise(data, size * sizeof(float), cudaMemAdviseSetReadMostly, device));
    
    // Başlat
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(i);
    }
    
    // Basit bir kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    auto sumKernel = [=] __device__ (int idx) {
        if (idx < size) {
            // Read-heavy operation
            float sum = 0.0f;
            for (int j = 0; j < 10; j++) {
                sum += data[idx];
            }
            data[idx] = sum / 10.0f;
        }
    };
    
    // Lambda kernel çalıştır
    auto kernel = [=] __global__ () {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        sumKernel(idx);
    };
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<gridSize, blockSize>>>();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float adviceTime;
    CUDA_CHECK(cudaEventElapsedTime(&adviceTime, start, stop));
    
    std::cout << "    With memory advice (ReadMostly): " << adviceTime << " ms" << std::endl;
    
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}