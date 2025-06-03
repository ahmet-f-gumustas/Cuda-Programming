#include "../include/common.h"

// Inclusive scan kernel
__global__ void inclusive_scan_kernel(const int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    // Load input into shared memory
    temp[tid] = (global_id < n) ? input[global_id] : 0;
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
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
    
    // Write result to output
    if (global_id < n) {
        output[global_id] = temp[tid];
    }
}

// Exclusive scan kernel
__global__ void exclusive_scan_kernel(const int* input, int* output, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    // Load input into shared memory
    temp[tid] = (global_id < n) ? input[global_id] : 0;
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Clear the last element for exclusive scan
    if (tid == 0) {
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
    
    // Write result to output
    if (global_id < n) {
        output[global_id] = temp[tid];
    }
}

// Segmented scan kernel
__global__ void segmented_scan_kernel(const int* input, const int* flags, 
                                     int* output, int n) {
    extern __shared__ int shared_data[];
    int* shared_values = shared_data;
    int* shared_flags = shared_data + blockDim.x;
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    // Load input and flags into shared memory
    if (global_id < n) {
        shared_values[tid] = input[global_id];
        shared_flags[tid] = flags[global_id];
    } else {
        shared_values[tid] = 0;
        shared_flags[tid] = 1; // Segment boundary
    }
    __syncthreads();
    
    // Segmented scan using flag propagation
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int next_val = 0;
        int next_flag = 0;
        
        if (tid >= stride) {
            next_val = shared_values[tid - stride];
            next_flag = shared_flags[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            if (!shared_flags[tid]) {
                shared_values[tid] += next_val;
            }
            shared_flags[tid] = shared_flags[tid] || next_flag;
        }
        __syncthreads();
    }
    
    // Write result to output
    if (global_id < n) {
        output[global_id] = shared_values[tid];
    }
}

// Inclusive scan implementation
void scan_inclusive_custom(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output.resize(n);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(int);
    
    inclusive_scan_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_output.get(), n);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    std::cout << "Custom Inclusive Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Exclusive scan implementation  
void scan_exclusive_custom(const std::vector<int>& input, std::vector<int>& output) {
    int n = input.size();
    output.resize(n);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = threads_per_block * sizeof(int);
    
    exclusive_scan_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_output.get(), n);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    std::cout << "Custom Exclusive Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Segmented scan implementation
void segmented_scan_custom(const std::vector<int>& input, const std::vector<int>& flags, 
                          std::vector<int>& output) {
    int n = input.size();
    output.resize(n);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_flags(n);
    ManagedMemory<int> d_output(n);
    
    d_input.copy_from_host(input.data());
    d_flags.copy_from_host(flags.data());
    
    CudaTimer timer;
    timer.start();
    
    int threads_per_block = BLOCK_SIZE;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = 2 * threads_per_block * sizeof(int); // For values and flags
    
    segmented_scan_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_input.get(), d_flags.get(), d_output.get(), n);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_output.copy_to_host(output.data());
    
    std::cout << "Custom Segmented Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Test function
void test_scan() {
    std::cout << "\n=== ADVANCED SCAN ALGORITHM TEST ===" << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 10);
    
    // Test inclusive scan
    std::vector<int> output_inclusive_custom, output_inclusive_thrust;
    scan_inclusive_custom(input, output_inclusive_custom);
    
    // Thrust comparison
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_output_thrust(n);
    
    CudaTimer timer;
    timer.start();
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output_thrust.begin());
    timer.stop();
    
    thrust::copy(d_output_thrust.begin(), d_output_thrust.end(), output_inclusive_thrust.begin());
    output_inclusive_thrust.resize(n);
    thrust::copy(d_output_thrust.begin(), d_output_thrust.end(), output_inclusive_thrust.begin());
    
    std::cout << "Thrust Inclusive Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Test exclusive scan
    std::vector<int> output_exclusive_custom, output_exclusive_thrust;
    scan_exclusive_custom(input, output_exclusive_custom);
    
    timer.start();
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output_thrust.begin());
    timer.stop();
    
    output_exclusive_thrust.resize(n);
    thrust::copy(d_output_thrust.begin(), d_output_thrust.end(), output_exclusive_thrust.begin());
    
    std::cout << "Thrust Exclusive Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Test segmented scan
    std::vector<int> flags(n, 0);
    // Create segment boundaries randomly
    for (int i = 0; i < n; i += rand() % 1000 + 100) {
        if (i < n) flags[i] = 1;
    }
    
    std::vector<int> output_segmented;
    segmented_scan_custom(input, flags, output_segmented);
    
    // Verify results (first 10 elements)
    std::cout << "\nVerification (first 10 elements):" << std::endl;
    std::cout << "Input:           ";
    for (int i = 0; i < 10; ++i) std::cout << input[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Inclusive Custom: ";
    for (int i = 0; i < 10; ++i) std::cout << output_inclusive_custom[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Inclusive Thrust: ";
    for (int i = 0; i < 10; ++i) std::cout << output_inclusive_thrust[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Exclusive Custom: ";
    for (int i = 0; i < 10; ++i) std::cout << output_exclusive_custom[i] << " ";
    std::cout << std::endl;
    
    std::cout << "Exclusive Thrust: ";
    for (int i = 0; i < 10; ++i) std::cout << output_exclusive_thrust[i] << " ";
    std::cout << std::endl;
    
    // Check correctness
    bool inclusive_match = (output_inclusive_custom == output_inclusive_thrust);
    bool exclusive_match = (output_exclusive_custom == output_exclusive_thrust);
    
    std::cout << "Inclusive scan results match: " << (inclusive_match ? "✓" : "✗") << std::endl;
    std::cout << "Exclusive scan results match: " << (exclusive_match ? "✓" : "✗") << std::endl;
}