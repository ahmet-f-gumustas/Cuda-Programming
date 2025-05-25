#include "matrix_benchmark.h"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

void initialize_matrix(Matrix& matrix, int size, bool random) {
    if (random) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (int i = 0; i < size * size; i++) {
            matrix[i] = dis(gen);
        }
    } else {
        // Test için basit desen
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                matrix[i * size + j] = static_cast<float>(i + j);
            }
        }
    }
}

bool verify_results(const Matrix& result1, const Matrix& result2, double tolerance) {
    if (result1.size() != result2.size()) {
        return false;
    }
    
    for (size_t i = 0; i < result1.size(); i++) {
        float diff = std::abs(result1[i] - result2[i]);
        float relative_error = diff / (std::abs(result1[i]) + 1e-7f);
        
        if (relative_error > tolerance) {
            return false;
        }
    }
    
    return true;
}

double calculate_max_error(const Matrix& result1, const Matrix& result2) {
    double max_error = 0.0;
    
    if (result1.size() != result2.size()) {
        return -1.0; // Hata durumu
    }
    
    for (size_t i = 0; i < result1.size(); i++) {
        double diff = std::abs(static_cast<double>(result1[i]) - static_cast<double>(result2[i]));
        max_error = std::max(max_error, diff);
    }
    
    return max_error;
}

DevicePtr allocate_device_memory(size_t size) {
    float* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    
    return DevicePtr(ptr, [](float* p) {
        if (p) {
            cudaFree(p);
        }
    });
}

void print_system_info() {
    std::cout << "Sistem Bilgileri:" << std::endl;
    
    // CUDA cihaz sayısı
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "  CUDA Cihaz Sayısı: " << device_count << std::endl;
    
    // Aktif cihaz bilgileri
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "  Aktif GPU: " << prop.name << std::endl;
    std::cout << "  CUDA Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bit" << std::endl;
    
    // Hesaplama kabiliyeti
    float memory_bandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
    std::cout << "  Teorik Memory Bandwidth: " << std::fixed << std::setprecision(1) 
              << memory_bandwidth << " GB/s" << std::endl;
}