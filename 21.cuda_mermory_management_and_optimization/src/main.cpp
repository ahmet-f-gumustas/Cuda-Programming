#include <iostream>
#include <vector>
#include "memory_manager.cuh"
#include "benchmark_utils.h"

int main(int argc, char** argv) {
    std::cout << "=== CUDA Memory Management and Optimization ===" << std::endl;
    
    // GPU özelliklerini yazdır
    printDeviceProperties();
    
    // Test edilecek veri boyutları
    std::vector<int> sizes = {1024, 1024*1024, 10*1024*1024, 100*1024*1024};
    
    for (int size : sizes) {
        std::cout << "\n=== Testing with size: " << size/1024.0/1024.0 << " MB ===" << std::endl;
        
        // Global Memory
        {
            BenchmarkTimer timer("Global Memory");
            runGlobalMemoryExample(size);
        }
        
        // Shared Memory (daha küçük boyutlar için)
        if (size <= 1024*1024) {
            BenchmarkTimer timer("Shared Memory");
            runSharedMemoryExample(size);
        }
        
        // Constant Memory
        {
            BenchmarkTimer timer("Constant Memory");
            runConstantMemoryExample(size);
        }
        
        // Unified Memory
        {
            BenchmarkTimer timer("Unified Memory");
            runUnifiedMemoryExample(size);
        }
    }
    
    return 0;
}