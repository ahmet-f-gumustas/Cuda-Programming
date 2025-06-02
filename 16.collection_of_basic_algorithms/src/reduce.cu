#include "../include/common.h"

// Naive reduce kernel
__global__ void naive_reduce_kernel(const int* input, int* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += input[i];
        }
        *output = sum;
    }
}

// Shared memory ile optimized reduce
__global__ void shared_reduce_kernel(const int* input, int* output, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared memory'ye veri yükle
    sdata[tid] = (global_id < n) ? input[global_id] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Blok sonucunu global memory'ye yaz
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Warp-level primitives kullanarak optimize edilmiş reduce
__global__ void warp_reduce_kernel(const int* input, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < n) ? input[tid] : 0;
    
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(__activemask(), val, offset);
    }
    
    // Her warp'ın ilk thread'i sonucu ekler
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(output, val);
    }
}

// İki aşamalı reduce (bloklar arası senkronizasyon olmadan)
__global__ void two_phase_reduce_kernel_step1(const int* input, int* block_sums, int n) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Shared memory'ye veri yükle
    sdata[tid] = (global_id < n) ? input[global_id] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Blok sonucunu kaydet
    if (tid == 0) {
        block_sums[bid] = sdata[0];
    }
}

__global__ void two_phase_reduce_kernel_step2(const int* block_sums, int* output, int num_blocks) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    
    // Blok toplamlarını shared memory'ye yükle
    sdata[tid] = (tid < num_blocks) ? block_sums[tid] : 0;
    __syncthreads();
    
    // Final reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_blocks) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *output = sdata[0];
    }
}

// Cooperative Groups kullanarak modern reduce
#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ int warp_reduce_sum(thread_group g, int val) {
    for (int offset = g.size() / 2; offset > 0; offset /= 2) {
        val += g.shfl_down(val, offset);
    }
    return val;
}

__global__ void coop_groups_reduce_kernel(const int* input, int* output, int n) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int tid = block.group_index().x * block.size() + block.thread_rank();
    int val = (tid < n) ? input[tid] : 0;
    
    // Warp-level reduction
    val = warp_reduce_sum(warp, val);
    
    // Shared memory for warp results
    extern __shared__ int warp_sums[];
    
    if (warp.thread_rank() == 0) {
        warp_sums[warp.meta_group_rank()] = val;
    }
    
    block.sync();
    
    // Final reduction of warp results
    if (warp.meta_group_rank() == 0) {
        val = (warp.thread_rank() < block.size() / warp.size()) ? 
              warp_sums[warp.thread_rank()] : 0;
        val = warp_reduce_sum(warp, val);
        
        if (warp.thread_rank() == 0) {
            atomicAdd(output, val);
        }
    }
}

// Ana reduce fonksiyonu
int reduce_custom(const std::vector<int>& input, const std::string& method) {
    int n = input.size();
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(1);
    
    d_input.copy_from_host(input.data());
    
    // Output'u sıfırla
    CUDA_CHECK(cudaMemset(d_output.get(), 0, sizeof(int)));
    
    CudaTimer timer;
    timer.start();
    
    if (method == "naive") {
        naive_reduce_kernel<<<1, 1>>>(d_input.get(), d_output.get(), n);
    }
    else if (method == "shared") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        int shared_mem_size = threads_per_block * sizeof(int);
        
        shared_reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_input.get(), d_output.get(), n);
    }
    else if (method == "warp") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        
        warp_reduce_kernel<<<blocks, threads_per_block>>>(
            d_input.get(), d_output.get(), n);
    }
    else if (method == "two_phase") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        
        ManagedMemory<int> d_block_sums(blocks);
        
        // İlk aşama: her blok kendi toplamını hesaplar
        two_phase_reduce_kernel_step1<<<blocks, threads_per_block, threads_per_block * sizeof(int)>>>(
            d_input.get(), d_block_sums.get(), n);
        
        CUDA_CHECK_KERNEL();
        
        // İkinci aşama: blok toplamlarını reduce et
        int final_threads = std::min(blocks, threads_per_block);
        two_phase_reduce_kernel_step2<<<1, final_threads, final_threads * sizeof(int)>>>(
            d_block_sums.get(), d_output.get(), blocks);
    }
    else if (method == "coop_groups") {
        int threads_per_block = BLOCK_SIZE;
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        int shared_mem_size = (threads_per_block / WARP_SIZE) * sizeof(int);
        
        coop_groups_reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_input.get(), d_output.get(), n);
    }
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    int result;
    d_output.copy_to_host(&result);
    
    std::cout << "Custom Reduce (" << method << ") - Time: " 
              << timer.elapsed_ms() << " ms, Result: " << result << std::endl;
    
    return result;
}

// Thrust ile karşılaştırma
int reduce_thrust(const std::vector<int>& input) {
    thrust::device_vector<int> d_input(input);
    
    CudaTimer timer;
    timer.start();
    
    int result = thrust::reduce(d_input.begin(), d_input.end(), 0, thrust::plus<int>());
    
    timer.stop();
    
    std::cout << "Thrust Reduce - Time: " << timer.elapsed_ms() 
              << " ms, Result: " << result << std::endl;
    
    return result;
}

// CUB ile karşılaştırma
int reduce_cub(const std::vector<int>& input) {
    int n = input.size();
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(1);
    
    d_input.copy_from_host(input.data());
    
    // Geçici storage boyutunu hesapla
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           d_input.get(), d_output.get(), n);
    
    // Geçici storage ayır
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    CudaTimer timer;
    timer.start();
    
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, 
                           d_input.get(), d_output.get(), n);
    
    timer.stop();
    
    int result;
    d_output.copy_to_host(&result);
    CUDA_CHECK(cudaFree(d_temp_storage));
    
    std::cout << "CUB Reduce - Time: " << timer.elapsed_ms() 
              << " ms, Result: " << result << std::endl;
    
    return result;
}

// Min/Max reduce variants
template<typename T>
__global__ void min_reduce_kernel(const T* input, T* output, int n) {
    extern __shared__ T sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Initialize with maximum value if out of bounds
    sdata[tid] = (global_id < n) ? input[global_id] : std::numeric_limits<T>::max();
    __syncthreads();
    
    // Reduction for minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMin(output, sdata[0]);
    }
}

template<typename T>
__global__ void max_reduce_kernel(const T* input, T* output, int n) {
    extern __shared__ T sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Initialize with minimum value if out of bounds
    sdata[tid] = (global_id < n) ? input[global_id] : std::numeric_limits<T>::lowest();
    __syncthreads();
    
    // Reduction for maximum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicMax(output, sdata[0]);
    }
}

// Test fonksiyonu
void test_reduce() {
    std::cout << "\n=== REDUCE ALGORITHM TEST ===" << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 10);
    
    // CPU referans sonucu
    long long cpu_sum = 0;
    for (int val : input) {
        cpu_sum += val;
    }
    
    std::cout << "CPU Reference Sum: " << cpu_sum << std::endl;
    
    // Custom implementasyonlar
    int result_naive = reduce_custom(input, "naive");
    int result_shared = reduce_custom(input, "shared");
    int result_warp = reduce_custom(input, "warp");
    int result_two_phase = reduce_custom(input, "two_phase");
    int result_coop = reduce_custom(input, "coop_groups");
    
    // Thrust ve CUB
    int result_thrust = reduce_thrust(input);
    int result_cub = reduce_cub(input);
    
    // Doğruluk kontrolü
    std::vector<int> results = {result_shared, result_warp, result_two_phase, 
                               result_coop, result_thrust, result_cub};
    
    bool all_match = true;
    for (int result : results) {
        if (result != cpu_sum) {
            all_match = false;
            break;
        }
    }
    
    std::cout << "Results match CPU: " << (all_match ? "✓" : "✗") << std::endl;
    
    // Min/Max test
    std::cout << "\nMin/Max Reduce Test:" << std::endl;
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_min_result(1);
    ManagedMemory<int> d_max_result(1);
    
    d_input.copy_from_host(input.data());
    
    // Initialize output values
    int init_min = std::numeric_limits<int>::max();
    int init_max = std::numeric_limits<int>::lowest();
    d_min_result.copy_from_host(&init_min);
    d_max_result.copy_from_host(&init_max);
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(int);
    
    min_reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_min_result.get(), n);
    max_reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_max_result.get(), n);
    
    CUDA_CHECK_KERNEL();
    
    int gpu_min, gpu_max;
    d_min_result.copy_to_host(&gpu_min);
    d_max_result.copy_to_host(&gpu_max);
    
    // CPU referans
    int cpu_min = *std::min_element(input.begin(), input.end());
    int cpu_max = *std::max_element(input.begin(), input.end());
    
    std::cout << "Min - CPU: " << cpu_min << ", GPU: " << gpu_min 
              << " " << (cpu_min == gpu_min ? "✓" : "✗") << std::endl;
    std::cout << "Max - CPU: " << cpu_max << ", GPU: " << gpu_max 
              << " " << (cpu_max == gpu_max ? "✓" : "✗") << std::endl;
}