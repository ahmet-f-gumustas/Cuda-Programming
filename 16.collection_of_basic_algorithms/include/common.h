#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <memory>
#include <string>
#include <iomanip>

// CUDA Error Checking Macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
    cudaDeviceSynchronize(); \
} while(0)

// Sabitler
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCKS = 65536;

// Timer sınıfı
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event, 0));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
    }
    
    float elapsed_ms() {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};

// Bellek yöneticisi
template<typename T>
class ManagedMemory {
private:
    T* d_ptr;
    size_t size;
    
public:
    ManagedMemory(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
    }
    
    ~ManagedMemory() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }
    
    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    size_t get_size() const { return size; }
    
    void copy_from_host(const T* h_ptr) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* h_ptr) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

// Yardımcı fonksiyonlar
template<typename T>
void print_device_array(const T* d_array, size_t n, const std::string& name) {
    std::vector<T> h_array(n);
    CUDA_CHECK(cudaMemcpy(h_array.data(), d_array, n * sizeof(T), cudaMemcpyDeviceToHost));
    
    std::cout << name << ": ";
    for (size_t i = 0; i < std::min(n, size_t(20)); ++i) {
        std::cout << h_array[i] << " ";
    }
    if (n > 20) std::cout << "...";
    std::cout << std::endl;
}

template<typename T>
std::vector<T> generate_random_data(size_t n, T min_val = 0, T max_val = 100) {
    std::vector<T> data(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dis(min_val, max_val);
        for (auto& val : data) {
            val = dis(gen);
        }
    } else {
        std::uniform_real_distribution<T> dis(min_val, max_val);
        for (auto& val : data) {
            val = dis(gen);
        }
    }
    
    return data;
}

// GPU bilgilerini yazdır
void print_gpu_info();

// Occupancy hesaplama
template<typename KernelFunc>
void print_occupancy_info(KernelFunc kernel, int block_size) {
    int min_grid_size, block_size_opt;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_opt, kernel, 0, 0));
    
    int max_active_blocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    float occupancy = (max_active_blocks * block_size / float(prop.maxThreadsPerMultiProcessor)) * 100;
    
    std::cout << "Kernel Occupancy Info:" << std::endl;
    std::cout << "  Optimal block size: " << block_size_opt << std::endl;
    std::cout << "  Current block size: " << block_size << std::endl;
    std::cout << "  Max active blocks per SM: " << max_active_blocks << std::endl;
    std::cout << "  Occupancy: " << std::fixed << std::setprecision(2) << occupancy << "%" << std::endl;
}