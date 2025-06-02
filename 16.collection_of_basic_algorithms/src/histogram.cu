#include "../include/common.h"

// Naive histogram kernel - atomik işlemler kullanarak
__global__ void naive_histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int bin = input[tid] % num_bins; // Basit mapping
        atomicAdd(&histogram[bin], 1);
    }
}

// Shared memory kullanarak optimize edilmiş histogram
__global__ void shared_histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared histogram'ı sıfırla
    for (int i = tid; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Her thread kendi elemanını işler
    if (global_id < n) {
        int bin = input[global_id] % num_bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Shared memory'den global memory'ye kopyala
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// Warp-level aggregation kullanarak optimize edilmiş histogram
__global__ void warp_histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    extern __shared__ int warp_hist[];
    int* my_warp_hist = &warp_hist[warp_id * num_bins];
    
    // Her warp kendi histogram alanını sıfırla
    for (int i = lane_id; i < num_bins; i += WARP_SIZE) {
        my_warp_hist[i] = 0;
    }
    __syncwarp();
    
    // Veri işle
    if (tid < n) {
        int bin = input[tid] % num_bins;
        atomicAdd(&my_warp_hist[bin], 1);
    }
    __syncwarp();
    
    // Warp sonuçlarını global histogram'a ekle
    for (int i = lane_id; i < num_bins; i += WARP_SIZE) {
        if (my_warp_hist[i] > 0) {
            atomicAdd(&histogram[i], my_warp_hist[i]);
        }
    }
}

// Privatization stratejisi - her thread kendi histogram'ını tutar
__global__ void privatized_histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int threads_per_block = blockDim.x;
    
    // Her thread'in kendi private histogram'ı
    int* my_hist = &shared_hist[tid * num_bins];
    
    // Private histogram'ı sıfırla
    for (int i = 0; i < num_bins; ++i) {
        my_hist[i] = 0;
    }
    __syncthreads();
    
    // Grid-stride loop ile veri işle
    for (int i = bid * threads_per_block + tid; i < n; i += gridDim.x * threads_per_block) {
        int bin = input[i] % num_bins;
        my_hist[bin]++;
    }
    __syncthreads();
    
    // Private histogram'ları birleştir
    for (int bin = 0; bin < num_bins; ++bin) {
        int count = my_hist[bin];
        if (count > 0) {
            atomicAdd(&histogram[bin], count);
        }
    }
}

// Coalesced access pattern ile optimize edilmiş histogram
__global__ void coalesced_histogram_kernel(const int* input, int* histogram, int n, int num_bins) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared histogram'ı sıfırla
    for (int i = tid; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Coalesced memory access pattern
    int stride = blockDim.x * gridDim.x;
    for (int i = global_id; i < n; i += stride) {
        int bin = input[i] % num_bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Sonuçları global memory'ye yaz
    for (int i = tid; i < num_bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// Multi-pass histogram (büyük bin sayıları için)
__global__ void multipass_histogram_kernel(const int* input, int* histogram, int n, 
                                          int num_bins, int pass, int bins_per_pass) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    int start_bin = pass * bins_per_pass;
    int end_bin = min(start_bin + bins_per_pass, num_bins);
    int local_bins = end_bin - start_bin;
    
    // Shared histogram'ı sıfırla
    for (int i = tid; i < local_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // Bu pass'e ait binleri işle
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x) {
        int bin = input[i] % num_bins;
        if (bin >= start_bin && bin < end_bin) {
            atomicAdd(&shared_hist[bin - start_bin], 1);
        }
    }
    __syncthreads();
    
    // Sonuçları global memory'ye yaz
    for (int i = tid; i < local_bins; i += blockDim.x) {
        if (shared_hist[i] > 0) {
            atomicAdd(&histogram[start_bin + i], shared_hist[i]);
        }
    }
}

// Ana histogram fonksiyonu
void histogram_custom(const std::vector<int>& input, std::vector<int>& histogram, 
                     int num_bins, const std::string& method) {
    int n = input.size();
    histogram.assign(num_bins, 0);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_histogram(num_bins);
    
    d_input.copy_from_host(input.data());
    CUDA_CHECK(cudaMemset(d_histogram.get(), 0, num_bins * sizeof(int)));
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    blocks = std::min(blocks, MAX_BLOCKS); // Grid boyutunu sınırla
    
    if (method == "naive") {
        naive_histogram_kernel<<<blocks, threads_per_block>>>(
            d_input.get(), d_histogram.get(), n, num_bins);
    }
    else if (method == "shared") {
        int shared_mem_size = num_bins * sizeof(int);
        
        // Shared memory sınırını kontrol et
        int max_shared_mem;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem, 
                   cudaDevAttrMaxSharedMemoryPerBlock, 0));
        
        if (shared_mem_size <= max_shared_mem) {
            shared_histogram_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        } else {
            std::cout << "Warning: Not enough shared memory, falling back to naive method" << std::endl;
            naive_histogram_kernel<<<blocks, threads_per_block>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        }
    }
    else if (method == "warp") {
        int warps_per_block = threads_per_block / WARP_SIZE;
        int shared_mem_size = warps_per_block * num_bins * sizeof(int);
        
        int max_shared_mem;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem, 
                   cudaDevAttrMaxSharedMemoryPerBlock, 0));
        
        if (shared_mem_size <= max_shared_mem) {
            warp_histogram_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        } else {
            std::cout << "Warning: Not enough shared memory for warp method" << std::endl;
            shared_histogram_kernel<<<blocks, threads_per_block, num_bins * sizeof(int)>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        }
    }
    else if (method == "privatized") {
        int shared_mem_size = threads_per_block * num_bins * sizeof(int);
        
        int max_shared_mem;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem, 
                   cudaDevAttrMaxSharedMemoryPerBlock, 0));
        
        if (shared_mem_size <= max_shared_mem) {
            privatized_histogram_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        } else {
            std::cout << "Warning: Not enough shared memory for privatized method" << std::endl;
            shared_histogram_kernel<<<blocks, threads_per_block, num_bins * sizeof(int)>>>(
                d_input.get(), d_histogram.get(), n, num_bins);
        }
    }
    else if (method == "coalesced") {
        int shared_mem_size = num_bins * sizeof(int);
        coalesced_histogram_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_input.get(), d_histogram.get(), n, num_bins);
    }
    else if (method == "multipass") {
        int max_shared_mem;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_mem, 
                   cudaDevAttrMaxSharedMemoryPerBlock, 0));
        
        int bins_per_pass = max_shared_mem / sizeof(int);
        int num_passes = (num_bins + bins_per_pass - 1) / bins_per_pass;
        
        for (int pass = 0; pass < num_passes; ++pass) {
            int shared_mem_size = std::min(bins_per_pass, num_bins - pass * bins_per_pass) * sizeof(int);
            multipass_histogram_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_input.get(), d_histogram.get(), n, num_bins, pass, bins_per_pass);
            CUDA_CHECK_KERNEL();
        }
    }
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_histogram.copy_to_host(histogram.data());
    
    std::cout << "Custom Histogram (" << method << ") - Time: " 
              << timer.elapsed_ms() << " ms" << std::endl;
}

// Thrust ile karşılaştırma
void histogram_thrust(const std::vector<int>& input, std::vector<int>& histogram, int num_bins) {
    histogram.assign(num_bins, 0);
    
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_keys(num_bins);
    thrust::device_vector<int> d_values(num_bins);
    
    // Bin anahtarlarını oluştur
    thrust::sequence(d_keys.begin(), d_keys.end());
    
    CudaTimer timer;
    timer.start();
    
    // Input değerlerini bin'lere map et
    thrust::device_vector<int> d_bins(input.size());
    thrust::transform(d_input.begin(), d_input.end(), d_bins.begin(), 
                     [num_bins] __device__ (int x) { return x % num_bins; });
    
    // Sort keys
    thrust::sort(d_bins.begin(), d_bins.end());
    
    // Reduce by key to count
    thrust::device_vector<int> d_unique_keys(num_bins);
    thrust::device_vector<int> d_counts(num_bins);
    
    auto end = thrust::reduce_by_key(d_bins.begin(), d_bins.end(),
                                    thrust::constant_iterator<int>(1),
                                    d_unique_keys.begin(),
                                    d_counts.begin());
    
    timer.stop();
    
    // Sonuçları host'a kopyala
    thrust::host_vector<int> h_unique_keys(d_unique_keys.begin(), end.first);
    thrust::host_vector<int> h_counts(d_counts.begin(), end.second);
    
    for (size_t i = 0; i < h_unique_keys.size(); ++i) {
        if (h_unique_keys[i] < num_bins) {
            histogram[h_unique_keys[i]] = h_counts[i];
        }
    }
    
    std::cout << "Thrust Histogram - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// CUB ile karşılaştırma
void histogram_cub(const std::vector<int>& input, std::vector<int>& histogram, int num_bins) {
    int n = input.size();
    histogram.assign(num_bins, 0);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_histogram(num_bins);
    
    d_input.copy_from_host(input.data());
    
    // Bin boundaries oluştur
    std::vector<int> bin_boundaries(num_bins + 1);
    for (int i = 0; i <= num_bins; ++i) {
        bin_boundaries[i] = i;
    }
    
    ManagedMemory<int> d_bin_boundaries(num_bins + 1);
    d_bin_boundaries.copy_from_host(bin_boundaries.data());
    
    // Geçici storage boyutunu hesapla
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Input değerlerini bin'lere map et
    thrust::device_vector<int> d_bins(n);
    thrust::transform(thrust::device_pointer_cast(d_input.get()),
                     thrust::device_pointer_cast(d_input.get()) + n,
                     d_bins.begin(),
                     [num_bins] __device__ (int x) { return x % num_bins; });
    
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                       thrust::raw_pointer_cast(d_bins.data()),
                                       d_histogram.get(), num_bins + 1, 0, num_bins, n);
    
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CudaTimer timer;
    timer.start();
    
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                       thrust::raw_pointer_cast(d_bins.data()),
                                       d_histogram.get(), num_bins + 1, 0, num_bins, n);
    
    timer.stop();
    
    d_histogram.copy_to_host(histogram.data());
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    std::cout << "CUB Histogram - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Test fonksiyonu
void test_histogram() {
    std::cout << "\n=== HISTOGRAM ALGORITHM TEST ===" << std::endl;
    
    const int n = 1000000;
    const int num_bins = 256;
    auto input = generate_random_data<int>(n, 0, num_bins - 1);
    
    // CPU referans
    std::vector<int> cpu_histogram(num_bins, 0);
    for (int val : input) {
        cpu_histogram[val % num_bins]++;
    }
    
    std::vector<int> hist_shared, hist_warp, hist_coalesced;
    std::vector<int> hist_thrust, hist_cub;
    
    // Custom implementasyonlar
    histogram_custom(input, hist_shared, num_bins, "shared");
    histogram_custom(input, hist_warp, num_bins, "warp");
    histogram_custom(input, hist_coalesced, num_bins, "coalesced");
    
    // Thrust ve CUB
    histogram_thrust(input, hist_thrust, num_bins);
    histogram_cub(input, hist_cub, num_bins);
    
    // Doğruluk kontrolü
    bool shared_match = (hist_shared == cpu_histogram);
    bool thrust_match = (hist_thrust == cpu_histogram);
    bool cub_match = (hist_cub == cpu_histogram);
    
    std::cout << "Shared memory result matches CPU: " << (shared_match ? "✓" : "✗") << std::endl;
    std::cout << "Thrust result matches CPU: " << (thrust_match ? "✓" : "✗") << std::endl;
    std::cout << "CUB result matches CPU: " << (cub_match ? "✓" : "✗") << std::endl;
    
    // İlk 10 bin'in sonuçlarını yazdır
    std::cout << "\nFirst 10 bins comparison:" << std::endl;
    std::cout << "Bin\tCPU\tShared\tThrust\tCUB" << std::endl;
    for (int i = 0; i < std::min(10, num_bins); ++i) {
        std::cout << i << "\t" << cpu_histogram[i] << "\t" << hist_shared[i] 
                  << "\t" << hist_thrust[i] << "\t" << hist_cub[i] << std::endl;
    }
}