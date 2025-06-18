#include "histogram_cuda.h"
#include <iostream>
#include <chrono>
#include <cstdio>

// CUDA hata kontrolü
void HistogramCUDA::checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d - %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Basit CUDA kernel - her thread bir veri noktasını işler
__global__ void histogramKernelBasic(
    const int* data, 
    int* histogram, 
    int data_size, 
    int num_bins, 
    int min_value, 
    int max_value
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < data_size) {
        int value = data[idx];
        
        // Değer aralık içindeyse
        if (value >= min_value && value <= max_value) {
            int range = max_value - min_value + 1;
            double bin_width = static_cast<double>(range) / num_bins;
            
            int bin_index = static_cast<int>((value - min_value) / bin_width);
            
            // Son bin için sınır kontrolü
            if (bin_index >= num_bins) {
                bin_index = num_bins - 1;
            }
            
            // Atomic operation kullanarak histogram güncelle
            atomicAdd(&histogram[bin_index], 1);
        }
    }
}

// Optimized CUDA kernel - shared memory kullanır
__global__ void histogramKernelOptimized(
    const int* data, 
    int* histogram, 
    int data_size, 
    int num_bins, 
    int min_value, 
    int max_value
) {
    // Shared memory histogram
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int idx = bid * block_size + tid;
    
    // Shared memory'yi sıfırla
    for (int i = tid; i < num_bins; i += block_size) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Her thread birden fazla eleman işleyebilir
    int range = max_value - min_value + 1;
    double bin_width = static_cast<double>(range) / num_bins;
    
    // Grid-stride loop pattern
    for (int i = idx; i < data_size; i += gridDim.x * block_size) {
        int value = data[i];
        
        if (value >= min_value && value <= max_value) {
            int bin_index = static_cast<int>((value - min_value) / bin_width);
            
            if (bin_index >= num_bins) {
                bin_index = num_bins - 1;
            }
            
            // Shared memory'de atomic add
            atomicAdd(&shared_hist[bin_index], 1);
        }
    }
    
    __syncthreads();
    
    // Shared memory'den global memory'ye kopyala
    for (int i = tid; i < num_bins; i += block_size) {
        atomicAdd(&histogram[i], shared_hist[i]);
    }
}

void HistogramCUDA::computeHistogramBasic(
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value
) {
    int data_size = data.size();
    
    // Device memory ayır
    int *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, data_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    
    // Veriyi device'a kopyala
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
    
    // Kernel parametreleri
    int block_size = 256;
    int grid_size = (data_size + block_size - 1) / block_size;
    
    // Kernel çalıştır
    histogramKernelBasic<<<grid_size, block_size>>>(
        d_data, d_histogram, data_size, num_bins, min_value, max_value
    );
    
    // Kernel tamamlanmasını bekle
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Sonucu host'a kopyala
    CUDA_CHECK(cudaMemcpy(histogram.data(), d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Memory'yi temizle
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));
}

void HistogramCUDA::computeHistogramOptimized(
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value
) {
    int data_size = data.size();
    
    // Device memory ayır
    int *d_data, *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_data, data_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    
    // Veriyi device'a kopyala
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
    
    // Kernel parametreleri
    int block_size = 256;
    int grid_size = std::min(32, (data_size + block_size - 1) / block_size); // Grid boyutunu sınırla
    
    // Shared memory boyutu
    int shared_mem_size = num_bins * sizeof(int);
    
    // Kernel çalıştır
    histogramKernelOptimized<<<grid_size, block_size, shared_mem_size>>>(
        d_data, d_histogram, data_size, num_bins, min_value, max_value
    );
    
    // Kernel tamamlanmasını bekle ve hata kontrolü
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Sonucu host'a kopyala
    CUDA_CHECK(cudaMemcpy(histogram.data(), d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Memory'yi temizle
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));
}

double HistogramCUDA::timeHistogramComputation(
    void (*compute_func)(const std::vector<int>&, std::vector<int>&, int, int, int),
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value
) {
    // CUDA event'ler ile daha hassas zamanlama
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    compute_func(data, histogram, num_bins, min_value, max_value);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return static_cast<double>(milliseconds);
}