#ifndef HISTOGRAM_CUDA_H
#define HISTOGRAM_CUDA_H

#include <vector>
#include <cuda_runtime.h>

class HistogramCUDA {
public:
    // CUDA ile basit histogram hesaplama
    static void computeHistogramBasic(
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255
    );
    
    // CUDA ile shared memory kullanarak optimized histogram
    static void computeHistogramOptimized(
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255
    );
    
    // Performans ölçümü
    static double timeHistogramComputation(
        void (*compute_func)(const std::vector<int>&, std::vector<int>&, int, int, int),
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255
    );
    
    // CUDA hata kontrolü yardımcı makrosu
    static void checkCudaError(cudaError_t error, const char* file, int line);
};

// CUDA hata kontrolü için makro
#define CUDA_CHECK(call) HistogramCUDA::checkCudaError(call, __FILE__, __LINE__)

// CUDA kernel fonksiyonları (device kodları)
__global__ void histogramKernelBasic(
    const int* data, 
    int* histogram, 
    int data_size, 
    int num_bins, 
    int min_value, 
    int max_value
);

__global__ void histogramKernelOptimized(
    const int* data, 
    int* histogram, 
    int data_size, 
    int num_bins, 
    int min_value, 
    int max_value
);

#endif // HISTOGRAM_CUDA_H