// ==================== src/merge_sort.cu ====================
#include "../include/common.h"

// Merge two sorted subarrays
__global__ void merge_kernel(int* data, int* temp, int left, int mid, int right) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid <= right - left) {
        int i = left;
        int j = mid + 1;
        int k = left;
        
        // Merge the two halves
        while (i <= mid && j <= right) {
            if (data[i] <= data[j]) {
                temp[k++] = data[i++];
            } else {
                temp[k++] = data[j++];
            }
        }
        
        // Copy remaining elements
        while (i <= mid) {
            temp[k++] = data[i++];
        }
        while (j <= right) {
            temp[k++] = data[j++];
        }
        
        // Copy back to original array
        for (int idx = left; idx <= right; ++idx) {
            data[idx] = temp[idx];
        }
    }
}

// Bottom-up merge sort implementation
void merge_sort_custom(std::vector<int>& data) {
    int n = data.size();
    
    ManagedMemory<int> d_data(n);
    ManagedMemory<int> d_temp(n);
    
    d_data.copy_from_host(data.data());
    
    CudaTimer timer;
    timer.start();
    
    // Bottom-up merge sort
    for (int size = 1; size < n; size *= 2) {
        for (int left = 0; left < n - 1; left += 2 * size) {
            int mid = std::min(left + size - 1, n - 1);
            int right = std::min(left + 2 * size - 1, n - 1);
            
            if (mid < right) {
                merge_kernel<<<1, 1>>>(d_data.get(), d_temp.get(), left, mid, right);
                CUDA_CHECK_KERNEL();
            }
        }
    }
    
    timer.stop();
    
    d_data.copy_to_host(data.data());
    
    std::cout << "Custom Merge Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Thrust implementation
void merge_sort_thrust(std::vector<int>& data) {
    thrust::device_vector<int> d_data(data);
    
    CudaTimer timer;
    timer.start();
    
    thrust::sort(d_data.begin(), d_data.end());
    
    timer.stop();
    
    thrust::copy(d_data.begin(), d_data.end(), data.begin());
    
    std::cout << "Thrust Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
}

// Test function
void test_merge_sort() {
    std::cout << "\n=== MERGE SORT TEST ===" << std::endl;
    
    const int n = 1000000;
    auto original_data = generate_random_data<int>(n, 0, 1000000);
    
    // CPU reference
    auto cpu_data = original_data;
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(cpu_data.begin(), cpu_data.end());
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    std::cout << "CPU std::sort - Time: " << cpu_time << " ms" << std::endl;
    
    // GPU implementations
    auto gpu_data_custom = original_data;
    auto gpu_data_thrust = original_data;
    
    merge_sort_custom(gpu_data_custom);
    merge_sort_thrust(gpu_data_thrust);
    
    // Verify results
    bool custom_correct = std::is_sorted(gpu_data_custom.begin(), gpu_data_custom.end());
    bool thrust_correct = std::is_sorted(gpu_data_thrust.begin(), gpu_data_thrust.end());
    bool custom_matches_cpu = (gpu_data_custom == cpu_data);
    bool thrust_matches_cpu = (gpu_data_thrust == cpu_data);
    
    std::cout << "\nResults verification:" << std::endl;
    std::cout << "Custom implementation is sorted: " << (custom_correct ? "✓" : "✗") << std::endl;
    std::cout << "Thrust implementation is sorted: " << (thrust_correct ? "✓" : "✗") << std::endl;
    std::cout << "Custom matches CPU: " << (custom_matches_cpu ? "✓" : "✗") << std::endl;
    std::cout << "Thrust matches CPU: " << (thrust_matches_cpu ? "✓" : "✗") << std::endl;
}
