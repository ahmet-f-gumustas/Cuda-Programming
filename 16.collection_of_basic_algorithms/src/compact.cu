#include "../include/common.h"

// Predicate evaluation kernel
__global__ void evaluate_predicate_kernel(const int* input, int* flags, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Example predicate: keep even numbers
        flags[tid] = (input[tid] % 2 == 0) ? 1 : 0;
    }
}

// Custom predicate kernel (with function parameter)
template<typename Predicate>
__global__ void evaluate_custom_predicate_kernel(const int* input, int* flags, int n, Predicate pred) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        flags[tid] = pred(input[tid]) ? 1 : 0;
    }
}

// Compact kernel using scan results
__global__ void compact_kernel(const int* input, const int* scan_result, 
                              const int* flags, int* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n && flags[tid]) {
        int output_index = scan_result[tid] - 1; // Convert to 0-based index
        output[output_index] = input[tid];
    }
}

// Work-efficient stream compaction
__global__ void work_efficient_compact_kernel(const int* input, int* output, 
                                             int* output_count, int n) {
    extern __shared__ int shared_data[];
    int* shared_input = shared_data;
    int* shared_flags = shared_data + blockDim.x;
    int* shared_scan = shared_data + 2 * blockDim.x;
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    // Load input and evaluate predicate
    if (global_id < n) {
        shared_input[tid] = input[global_id];
        shared_flags[tid] = (input[global_id] % 2 == 0) ? 1 : 0; // Even numbers
    } else {
        shared_input[tid] = 0;
        shared_flags[tid] = 0;
    }
    __syncthreads();
    
    // Perform exclusive scan on flags
    shared_scan[tid] = shared_flags[tid];
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared_scan[index] += shared_scan[index - stride];
        }
        __syncthreads();
    }
    
    // Clear last element for exclusive scan
    if (tid == 0) {
        shared_scan[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = shared_scan[index];
            shared_scan[index] += shared_scan[index - stride];
            shared_scan[index - stride] = temp;
        }
        __syncthreads();
    }
    
    // Compact the elements
    if (global_id < n && shared_flags[tid]) {
        int output_pos = shared_scan[tid] + blockIdx.x * blockDim.x; // Global position
        output[output_pos] = shared_input[tid];
        atomicAdd(output_count, 1);
    }
}

// Warp-level compaction
__global__ void warp_compact_kernel(const int* input, int* output, int* count, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (tid >= n) return;
    
    int value = input[tid];
    bool predicate = (value % 2 == 0); // Even numbers
    
    // Warp vote to get mask of valid elements
    unsigned int mask = __ballot_sync(__activemask(), predicate);
    
    if (predicate) {
        // Count preceding valid elements in warp
        unsigned int preceding_mask = mask & ((1u << lane_id) - 1);
        int warp_offset = __popc(preceding_mask);
        
        // Get global offset for this warp
        int global_offset = 0;
        if (lane_id == 0) {
            global_offset = atomicAdd(count, __popc(mask));
        }
        global_offset = __shfl_sync(__activemask(), global_offset, 0);
        
        // Write to output
        output[global_offset + warp_offset] = value;
    }
}

// Custom compact implementation
void compact_custom(const std::vector<int>& input, std::vector<int>& output, 
                   std::function<bool(int)> predicate) {
    int n = input.size();
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_flags(n);
    ManagedMemory<int> d_scan_result(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // Step 1: Evaluate predicate
    evaluate_predicate_kernel<<<blocks, threads_per_block>>>(
        d_input.get(), d_flags.get(), n);
    CUDA_CHECK_KERNEL();
    
    // Step 2: Exclusive scan on flags
    thrust::device_ptr<int> thrust_flags(d_flags.get());
    thrust::device_ptr<int> thrust_scan(d_scan_result.get());
    thrust::exclusive_scan(thrust_flags, thrust_flags + n, thrust_scan);
    
    // Step 3: Compact elements
    compact_kernel<<<blocks, threads_per_block>>>(
        d_input.get(), d_scan_result.get(), d_flags.get(), d_output.get(), n);
    CUDA_CHECK_KERNEL();
    
    timer.stop();
    
    // Get final count
    std::vector<int> h_flags(n);
    d_flags.copy_to_host(h_flags.data());
    int final_count = std::accumulate(h_flags.begin(), h_flags.end(), 0);
    
    // Copy result
    output.resize(final_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), 
                         final_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Custom Stream Compaction - Time: " << timer.elapsed_ms() 
              << " ms, Output size: " << final_count << std::endl;
}

// Thrust implementation
void compact_thrust(const std::vector<int>& input, std::vector<int>& output,
                   std::function<bool(int)> predicate) {
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_output(input.size());
    
    CudaTimer timer;
    timer.start();
    
    // Use thrust::copy_if
    auto end = thrust::copy_if(d_input.begin(), d_input.end(), d_output.begin(),
                              [=] __device__ (int x) { return x % 2 == 0; });
    
    timer.stop();
    
    int final_count = end - d_output.begin();
    output.resize(final_count);
    thrust::copy(d_output.begin(), end, output.begin());
    
    std::cout << "Thrust Stream Compaction - Time: " << timer.elapsed_ms() 
              << " ms, Output size: " << final_count << std::endl;
}

// Work-efficient implementation
void compact_work_efficient(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    ManagedMemory<int> d_count(1);
    
    d_input.copy_from_host(input.data());
    CUDA_CHECK(cudaMemset(d_count.get(), 0, sizeof(int)));
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = 3 * threads_per_block * sizeof(int);
    
    work_efficient_compact_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_output.get(), d_count.get(), n);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    // Get final count
    int final_count;
    d_count.copy_to_host(&final_count);
    
    output.resize(final_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), 
                         final_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Work-efficient Compaction - Time: " << timer.elapsed_ms() 
              << " ms, Output size: " << final_count << std::endl;
}

// Warp-level implementation
void compact_warp_level(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    ManagedMemory<int> d_count(1);
    
    d_input.copy_from_host(input.data());
    CUDA_CHECK(cudaMemset(d_count.get(), 0, sizeof(int)));
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    warp_compact_kernel<<<blocks, threads_per_block>>>(
        d_input.get(), d_output.get(), d_count.get(), n);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    // Get final count
    int final_count;
    d_count.copy_to_host(&final_count);
    
    output.resize(final_count);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), 
                         final_count * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "Warp-level Compaction - Time: " << timer.elapsed_ms() 
              << " ms, Output size: " << final_count << std::endl;
}

// Test function
void test_compact() {
    std::cout << "\n=== STREAM COMPACTION TEST ===" << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 100);
    
    // Predicate: keep even numbers
    auto even_predicate = [](int x) { return x % 2 == 0; };
    
    // CPU reference
    std::vector<int> cpu_result;
    auto start = std::chrono::high_resolution_clock::now();
    std::copy_if(input.begin(), input.end(), std::back_inserter(cpu_result), even_predicate);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    std::cout << "CPU Stream Compaction - Time: " << cpu_time 
              << " ms, Output size: " << cpu_result.size() << std::endl;
    
    // GPU implementations
    std::vector<int> gpu_result_custom, gpu_result_thrust, gpu_result_work_efficient, gpu_result_warp;
    
    compact_custom(input, gpu_result_custom, even_predicate);
    compact_thrust(input, gpu_result_thrust, even_predicate);
    compact_work_efficient(input, gpu_result_work_efficient);
    compact_warp_level(input, gpu_result_warp);
    
    // Verify results
    bool custom_match = (gpu_result_custom.size() == cpu_result.size());
    bool thrust_match = (gpu_result_thrust.size() == cpu_result.size());
    bool work_efficient_match = (gpu_result_work_efficient.size() == cpu_result.size());
    bool warp_match = (gpu_result_warp.size() == cpu_result.size());
    
    if (custom_match) {
        custom_match = std::equal(gpu_result_custom.begin(), gpu_result_custom.end(), cpu_result.begin());
    }
    if (thrust_match) {
        thrust_match = std::equal(gpu_result_thrust.begin(), gpu_result_thrust.end(), cpu_result.begin());
    }
    
    std::cout << "\nResults verification:" << std::endl;
    std::cout << "Custom implementation matches CPU: " << (custom_match ? "✓" : "✗") << std::endl;
    std::cout << "Thrust implementation matches CPU: " << (thrust_match ? "✓" : "✗") << std::endl;
    std::cout << "Work-efficient matches CPU: " << (work_efficient_match ? "✓" : "✗") << std::endl;
    std::cout << "Warp-level matches CPU: " << (warp_match ? "✓" : "✗") << std::endl;
    
    // Statistics
    double compression_ratio = (double)cpu_result.size() / input.size() * 100.0;
    std::cout << "\nCompression statistics:" << std::endl;
    std::cout << "Original size: " << input.size() << std::endl;
    std::cout << "Compacted size: " << cpu_result.size() << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(1) 
              << compression_ratio << "%" << std::endl;
    
    // Show first 20 elements of input and output
    std::cout << "\nFirst 20 elements:" << std::endl;
    std::cout << "Input:  ";
    for (int i = 0; i < std::min(20, (int)input.size()); ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output: ";
    for (int i = 0; i < std::min(20, (int)cpu_result.size()); ++i) {
        std::cout << cpu_result[i] << " ";
    }
    std::cout << std::endl;
}