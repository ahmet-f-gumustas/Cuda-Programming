#include "memory_manager.cuh"
#include <cuda_runtime.h>

void printDeviceProperties() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\n=== Device " << i << ": " << prop.name << " ===" << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Constant memory: " << prop.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    }
}

float measureKernelTime(void (*kernel)(int), int size, int iterations) {
    // Warm-up
    kernel(size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Ölçüm
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iterations; i++) {
        kernel(size);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds / iterations;
}