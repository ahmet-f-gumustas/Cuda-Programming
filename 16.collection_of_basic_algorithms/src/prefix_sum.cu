#include "../include/common.h"

// Naive prefix sum kernel (her thread bir eleman için)
__global__ void naive_prefix_sum_kernel(const int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    int sum = 0;
    for (int i = 0; i <= tid; ++i) {
        sum += input[i];
    }
    output[tid] = sum;
}

// Blok seviyesinde shared memory kullanan prefix sum
__global__ void block_prefix_sum_kernel(const int* input, int* output, int* block_sums, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared memory'ye veri yükle
    if (global_id < n) {
        temp[tid] = input[global_id];
    } else {
        temp[tid] = 0;
    }
    
    __syncthreads();
    
    // Up-sweep (reduction) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Blok toplamını kaydet
    if (tid == 0 && block_sums) {
        block_sums[bid] = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0;
    }
    
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp_val = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = temp_val;
        }
        __syncthreads();
    }
    
    // Sonucu global memory'ye yaz
    if (global_id < n) {
        output[global_id] = temp[tid];
    }
}

// Blok toplamlarını ekleyen kernel
__global__ void add_block_sums_kernel(int* output, const int* block_sums, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n || blockIdx.x == 0) return;
    
    // Bu bloktan önceki tüm blokların toplamını ekle
    int block_sum = 0;
    for (int i = 0; i < blockIdx.x; ++i) {
        block_sum += block_sums[i];
    }
    
    output[tid] += block_sum;
}

// Warp-level primitive kullanarak optimize edilmiş prefix sum
__global__ void warp_prefix_sum_kernel(const int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    int val = input[tid];
    
    // Warp-level prefix sum
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int temp = __shfl_up_sync(__activemask(), val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val += temp;
        }
    }
    
    output[tid] = val;
}

// Ana prefix sum fonksiyonu
void prefix_sum_custom(const std::vector<int>& input, std::vector<int>& output, const std::string& method) {
    int n = input.size();
    output.resize(n);
    
    // Device memory ayır
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    
    CudaTimer timer;
    timer.start();
    
    if (method == "naive") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        
        naive_prefix_sum_kernel<<<blocks, threads_per_block>>>(
            d_input.get(), d_output.get(), n);
    }
    else if (method == "shared") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        int shared_mem_size = threads_per_block * sizeof(int);
        
        // İki aşamalı: önce blok içi, sonra bloklar arası
        ManagedMemory<int> d_block_sums(blocks);
        
        block_prefix_sum_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_input.get(), d_output.get(), d_block_sums.get(), n);
        
        CUDA_CHECK_KERNEL();
        
        if (blocks > 1) {
            // Blok toplamlarının kendi prefix sum'ını hesapla
            std::vector<int> h_block_sums(blocks);
            d_block_sums.copy_to_host(h_block_sums.data());
            
            for (int i = 1; i < blocks; ++i) {
                h_block_sums[i] += h_block_sums[i-1];
            }
            
            d_block_sums.copy_from_host(h_block_sums.data());
            
            // Blok toplamlarını ekle
            add_block_sums_kernel<<<blocks, threads_per_block>>>(
                d_output.get(), d_block_sums.get(), n);
        }
    }
    else if (method == "warp") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        
        warp_prefix_sum_kernel<<<blocks, threads_per_block>>>(
            d_input.get(), d_output.get(), n);
    }
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    std::cout << "Custom Prefix Sum (" << method << ") - Time: " 
              << timer.elapsed_ms() << " ms" << std::endl;
}

// Thrust ile karşılaştırma
void prefix_sum_thrust(const std::vector<int>& input, std::vector<int>& output) {
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_output(input.size());
    
    CudaTimer timer;
    timer.start();
    
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    
    timer.stop();
    
    thrust::copy(d_output.begin(), d_output.end(), output.begin());
    
    std::cout << "Thrust Prefix Sum - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// CUB ile karşılaştırma
void prefix_sum_cub(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output.resize(n);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    
    // Geçici storage boyutunu hesapla
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, 
                                  d_input.get(), d_output.get(), n);
    
    // Geçici storage ayır
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CudaTimer timer;
    timer.start();
    
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, 
                                  d_input.get(), d_output.get(), n);
    
    timer.stop();
    
    d_output.copy_to_host(output.data());
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    std::cout << "CUB Prefix Sum - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Test fonksiyonu
void test_prefix_sum() {
    std::cout << "\n=== PREFIX SUM ALGORITHM TEST ===" << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 10);
    
    std::vector<int> output_custom_naive, output_custom_shared, output_custom_warp;
    std::vector<int> output_thrust, output_cub;
    
    // Custom implementasyonlar
    prefix_sum_custom(input, output_custom_naive, "naive");
    prefix_sum_custom(input, output_custom_shared, "shared");
    prefix_sum_custom(input, output_custom_warp, "warp");
    
    // Thrust ve CUB
    output_thrust.resize(n);
    output_cub.resize(n);
    prefix_sum_thrust(input, output_thrust);
    prefix_sum_cub(input, output_cub);
    
    // Doğruluk kontrolü (küçük örnek ile)
    if (n >= 10) {
        std::cout << "\nVerification (first 10 elements):" << std::endl;
        std::cout << "Input:  ";
        for (int i = 0; i < 10; ++i) std::cout << input[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Shared: ";
        for (int i = 0; i < 10; ++i) std::cout << output_custom_shared[i] << " ";
        std::cout << std::endl;
        
        std::cout << "Thrust: ";
        for (int i = 0; i < 10; ++i) std::cout << output_thrust[i] << " ";
        std::cout << std::endl;
    }
    
    // Sonuçları karşılaştır
    bool match = true;
    for (int i = 0; i < n && match; ++i) {
        if (output_custom_shared[i] != output_thrust[i] || 
            output_thrust[i] != output_cub[i]) {
            match = false;
        }
    }
    
    std::cout << "Results match: " << (match ? "✓" : "✗") << std::endl;
}