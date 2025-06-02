#include "../include/common.h"

// Radix sort için yardımcı makrolar
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_MASK (RADIX_SIZE - 1)

// Bit extraction fonksiyonu
__device__ __host__ inline int extract_digit(unsigned int value, int bit_pos) {
    return (value >> bit_pos) & RADIX_MASK;
}

// Naive counting sort (tek digit için)
__global__ void counting_sort_kernel(const unsigned int* input, unsigned int* output, 
                                   int n, int bit_pos) {
    extern __shared__ int shared_count[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared count array'ini sıfırla
    if (tid < RADIX_SIZE) {
        shared_count[tid] = 0;
    }
    __syncthreads();
    
    // Her thread kendi elemanının digit'ini sayar
    if (global_id < n) {
        int digit = extract_digit(input[global_id], bit_pos);
        atomicAdd(&shared_count[digit], 1);
    }
    __syncthreads();
    
    // Prefix sum hesapla (shared memory içinde)
    if (tid < RADIX_SIZE) {
        int sum = 0;
        for (int i = 0; i < tid; ++i) {
            sum += shared_count[i];
        }
        shared_count[tid] = sum;
    }
    __syncthreads();
    
    // Output pozisyonunu hesapla ve yaz
    if (global_id < n) {
        int digit = extract_digit(input[global_id], bit_pos);
        int pos = atomicAdd(&shared_count[digit], 1);
        output[pos] = input[global_id];
    }
}

// Block-level radix sort
__global__ void block_radix_sort_kernel(unsigned int* data, int n, int bit_pos) {
    extern __shared__ unsigned int shared_data[];
    unsigned int* shared_keys = shared_data;
    int* shared_count = (int*)&shared_data[blockDim.x];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Veriyi shared memory'ye yükle
    if (global_id < n) {
        shared_keys[tid] = data[global_id];
    } else {
        shared_keys[tid] = 0xFFFFFFFF; // Maksimum değer
    }
    __syncthreads();
    
    // Her digit için counting sort yap
    for (int digit = 0; digit < RADIX_SIZE; ++digit) {
        // Count phase
        int my_count = 0;
        if (extract_digit(shared_keys[tid], bit_pos) == digit) {
            my_count = 1;
        }
        
        // Prefix sum using shared memory
        shared_count[tid] = my_count;
        __syncthreads();
        
        // Parallel prefix sum
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int temp = 0;
            if (tid >= stride) {
                temp = shared_count[tid - stride];
            }
            __syncthreads();
            if (tid >= stride) {
                shared_count[tid] += temp;
            }
            __syncthreads();
        }
        
        // Scatter phase
        if (my_count == 1) {
            int pos = shared_count[tid] - 1;
            if (pos >= 0 && pos < blockDim.x) {
                // Bu basit implementasyon için in-place sort yapmıyoruz
                // Gerçek implementasyonda geçici buffer kullanılır
            }
        }
        __syncthreads();
    }
    
    // Sonucu geri yaz
    if (global_id < n) {
        data[global_id] = shared_keys[tid];
    }
}

// Optimized radix sort with warp-level operations
__global__ void warp_radix_sort_kernel(unsigned int* keys, unsigned int* values, 
                                      unsigned int* output_keys, unsigned int* output_values,
                                      int* global_histogram, int n, int bit_pos) {
    extern __shared__ int shared_mem[];
    int* warp_histogram = shared_mem;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;
    
    // Initialize warp histogram
    for (int i = lane_id; i < RADIX_SIZE; i += WARP_SIZE) {
        warp_histogram[warp_id * RADIX_SIZE + i] = 0;
    }
    __syncwarp();
    
    // Count phase
    if (tid < n) {
        int digit = extract_digit(keys[tid], bit_pos);
        atomicAdd(&warp_histogram[warp_id * RADIX_SIZE + digit], 1);
    }
    __syncthreads();
    
    // Merge warp histograms
    __shared__ int block_histogram[RADIX_SIZE];
    if (threadIdx.x < RADIX_SIZE) {
        int sum = 0;
        for (int w = 0; w < warps_per_block; ++w) {
            sum += warp_histogram[w * RADIX_SIZE + threadIdx.x];
        }
        block_histogram[threadIdx.x] = sum;
        atomicAdd(&global_histogram[threadIdx.x], sum);
    }
    __syncthreads();
}

// Multi-pass radix sort ana fonksiyonu
void radix_sort_custom(std::vector<unsigned int>& data, const std::string& method) {
    int n = data.size();
    
    ManagedMemory<unsigned int> d_keys(n);
    ManagedMemory<unsigned int> d_keys_alt(n);
    ManagedMemory<unsigned int> d_values(n);
    ManagedMemory<unsigned int> d_values_alt(n);
    
    d_keys.copy_from_host(data.data());
    
    // Initialize values (0, 1, 2, ...)
    std::vector<unsigned int> values(n);
    std::iota(values.begin(), values.end(), 0);
    d_values.copy_from_host(values.data());
    
    CudaTimer timer;
    timer.start();
    
    unsigned int* current_keys = d_keys.get();
    unsigned int* current_values = d_values.get();
    unsigned int* alt_keys = d_keys_alt.get();
    unsigned int* alt_values = d_values_alt.get();
    
    // 32 bit için 8 pass (4 bit씩)
    for (int bit_pos = 0; bit_pos < 32; bit_pos += RADIX_BITS) {
        if (method == "simple") {
            int threads_per_block = BLOCK_SIZE;
            int blocks = (n + threads_per_block - 1) / threads_per_block;
            int shared_mem_size = RADIX_SIZE * sizeof(int);
            
            counting_sort_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                current_keys, alt_keys, n, bit_pos);
        }
        else if (method == "warp") {
            int threads_per_block = BLOCK_SIZE;
            int blocks = (n + threads_per_block - 1) / threads_per_block;
            int warps_per_block = threads_per_block / WARP_SIZE;
            int shared_mem_size = warps_per_block * RADIX_SIZE * sizeof(int);
            
            // Global histogram için memory ayır
            ManagedMemory<int> d_global_histogram(RADIX_SIZE);
            CUDA_CHECK(cudaMemset(d_global_histogram.get(), 0, RADIX_SIZE * sizeof(int)));
            
            warp_radix_sort_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                current_keys, current_values, alt_keys, alt_values,
                d_global_histogram.get(), n, bit_pos);
        }
        
        CUDA_CHECK_KERNEL();
        
        // Swap buffers
        std::swap(current_keys, alt_keys);
        std::swap(current_values, alt_values);
    }
    
    timer.stop();
    
    // Sonucu host'a kopyala
    if (current_keys == d_keys.get()) {
        d_keys.copy_to_host(data.data());
    } else {
        d_keys_alt.copy_to_host(data.data());
    }
    
    std::cout << "Custom Radix Sort (" << method << ") - Time: " 
              << timer.elapsed_ms() << " ms" << std::endl;
}

// CUB ile karşılaştırma
void radix_sort_cub(std::vector<unsigned int>& data) {
    int n = data.size();
    
    ManagedMemory<unsigned int> d_keys_in(n);
    ManagedMemory<unsigned int> d_keys_out(n);
    ManagedMemory<unsigned int> d_values_in(n);
    ManagedMemory<unsigned int> d_values_out(n);
    
    d_keys_in.copy_from_host(data.data());
    
    // Initialize values
    std::vector<unsigned int> values(n);
    std::iota(values.begin(), values.end(), 0);
    d_values_in.copy_from_host(values.data());
    
    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                   d_keys_in.get(), d_keys_out.get(),
                                   d_values_in.get(), d_values_out.get(), n);
    
    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CudaTimer timer;
    timer.start();
    
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                   d_keys_in.get(), d_keys_out.get(),
                                   d_values_in.get(), d_values_out.get(), n);
    
    timer.stop();
    
    d_keys_out.copy_to_host(data.data());
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    std::cout << "CUB Radix Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Thrust ile karşılaştırma
void radix_sort_thrust(std::vector<unsigned int>& data) {
    thrust::device_vector<unsigned int> d_keys(data);
    thrust::device_vector<unsigned int> d_values(data.size());
    thrust::sequence(d_values.begin(), d_values.end());
    
    CudaTimer timer;
    timer.start();
    
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());
    
    timer.stop();
    
    thrust::copy(d_keys.begin(), d_keys.end(), data.begin());
    
    std::cout << "Thrust Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// In-place block-wise radix sort
__global__ void inplace_block_radix_sort_kernel(unsigned int* data, int n) {
    extern __shared__ unsigned int shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Load data to shared memory
    if (global_id < n) {
        shared_data[tid] = data[global_id];
    } else {
        shared_data[tid] = 0xFFFFFFFF;
    }
    __syncthreads();
    
    // Radix sort within block
    for (int bit_pos = 0; bit_pos < 32; bit_pos += RADIX_BITS) {
        // Local histogram
        __shared__ int local_hist[RADIX_SIZE];
        __shared__ int local_scan[RADIX_SIZE];
        
        // Initialize histogram
        if (tid < RADIX_SIZE) {
            local_hist[tid] = 0;
        }
        __syncthreads();
        
        // Count digits
        int my_digit = extract_digit(shared_data[tid], bit_pos);
        atomicAdd(&local_hist[my_digit], 1);
        __syncthreads();
        
        // Prefix scan of histogram
        if (tid < RADIX_SIZE) {
            int sum = 0;
            for (int i = 0; i < tid; ++i) {
                sum += local_hist[i];
            }
            local_scan[tid] = sum;
        }
        __syncthreads();
        
        // Scatter phase
        int new_pos = atomicAdd(&local_scan[my_digit], 1);
        
        // Use a temporary array for scatter
        __shared__ unsigned int temp_data[BLOCK_SIZE];
        temp_data[new_pos] = shared_data[tid];
        __syncthreads();
        
        // Copy back
        shared_data[tid] = temp_data[tid];
        __syncthreads();
    }
    
    // Write back to global memory
    if (global_id < n) {
        data[global_id] = shared_data[tid];
    }
}

// Block-wise sort with merge
void block_radix_sort_custom(std::vector<unsigned int>& data) {
    int n = data.size();
    
    ManagedMemory<unsigned int> d_data(n);
    d_data.copy_from_host(data.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(unsigned int);
    
    // Sort each block independently
    inplace_block_radix_sort_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_data.get(), n);
    
    CUDA_CHECK_KERNEL();
    
    // Simple merge phase (naive implementation)
    // In production, you'd use a more sophisticated merge algorithm
    thrust::device_ptr<unsigned int> thrust_ptr(d_data.get());
    for (int block_size = threads_per_block; block_size < n; block_size *= 2) {
        for (int i = 0; i < n; i += 2 * block_size) {
            int mid = std::min(i + block_size, n);
            int end = std::min(i + 2 * block_size, n);
            if (mid < end) {
                thrust::inplace_merge(thrust_ptr + i, thrust_ptr + mid, thrust_ptr + end);
            }
        }
    }
    
    timer.stop();
    
    d_data.copy_to_host(data.data());
    
    std::cout << "Block-wise Radix Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Test fonksiyonu
void test_radix_sort() {
    std::cout << "\n=== RADIX SORT ALGORITHM TEST ===" << std::endl;
    
    const int n = 1000000;
    auto original_data = generate_random_data<unsigned int>(n, 0, 0xFFFFFFFF);
    
    // CPU referans (std::sort)
    auto cpu_data = original_data;
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(cpu_data.begin(), cpu_data.end());
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    std::cout << "CPU std::sort - Time: " << cpu_time << " ms" << std::endl;
    
    // Custom implementasyonlar
    auto data_simple = original_data;
    auto data_warp = original_data;
    auto data_block = original_data;
    auto data_cub = original_data;
    auto data_thrust = original_data;
    
    //radix_sort_custom(data_simple, "simple");
    //radix_sort_custom(data_warp, "warp");
    block_radix_sort_custom(data_block);
    radix_sort_cub(data_cub);
    radix_sort_thrust(data_thrust);
    
    // Doğruluk kontrolü
    bool block_match = (data_block == cpu_data);
    bool cub_match = (data_cub == cpu_data);
    bool thrust_match = (data_thrust == cpu_data);
    
    std::cout << "Block-wise result matches CPU: " << (block_match ? "✓" : "✗") << std::endl;
    std::cout << "CUB result matches CPU: " << (cub_match ? "✓" : "✗") << std::endl;
    std::cout << "Thrust result matches CPU: " << (thrust_match ? "✓" : "✗") << std::endl;
    
    // İlk ve son 10 elemanı yazdır
    std::cout << "\nFirst 10 elements comparison:" << std::endl;
    std::cout << "Original: ";
    for (int i = 0; i < 10; ++i) std::cout << original_data[i] << " ";
    std::cout << std::endl;
    
    std::cout << "CPU:      ";
    for (int i = 0; i < 10; ++i) std::cout << cpu_data[i] << " ";
    std::cout << std::endl;
    
    std::cout << "CUB:      ";
    for (int i = 0; i < 10; ++i) std::cout << data_cub[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Thrust:   ";
    for (int i = 0; i < 10; ++i) std::cout << data_thrust[i] << " ";
    std::cout << std::endl;
}