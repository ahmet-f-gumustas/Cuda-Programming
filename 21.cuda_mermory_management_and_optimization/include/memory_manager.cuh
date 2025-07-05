#ifndef MEMORY_MANAGER_CUH
#define MEMORY_MANAGER_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// CUDA hata kontrolü makrosu
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while(0)

// Bellek türleri için örnek fonksiyonlar
void runGlobalMemoryExample(int size);
void runSharedMemoryExample(int size);
void runConstantMemoryExample(int size);
void runUnifiedMemoryExample(int size);

// Yardımcı fonksiyonlar
void printDeviceProperties();
float measureKernelTime(void (*kernel)(int), int size, int iterations = 100);

#endif // MEMORY_MANAGER_CUH