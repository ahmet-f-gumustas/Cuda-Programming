#include "../include/common.h"

// Test fonksiyonlarını declare et
void test_prefix_sum();
void test_reduce();
void test_histogram();
void test_radix_sort();
void test_bfs();
void test_scan();
void test_compact();
void test_matrix_multiply();
void test_merge_sort();
void test_convolution();

// GPU bilgilerini yazdır
void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Block Dimensions: (" << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Clock Rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
    std::cout << "========================\n" << std::endl;
}

// Menü göster
void show_menu() {
    std::cout << "\n=== PARALLEL ALGORITHMS COLLECTION ===" << std::endl;
    std::cout << "1.  Prefix Sum (Scan)" << std::endl;
    std::cout << "2.  Reduce" << std::endl;
    std::cout << "3.  Histogram" << std::endl;
    std::cout << "4.  Radix Sort" << std::endl;
    std::cout << "5.  BFS (Breadth-First Search)" << std::endl;
    std::cout << "6.  Scan (Advanced)" << std::endl;
    std::cout << "7.  Compact (Stream Compaction)" << std::endl;
    std::cout << "8.  Matrix Multiplication" << std::endl;
    std::cout << "9.  Merge Sort" << std::endl;
    std::cout << "10. Convolution" << std::endl;
    std::cout << "11. Run All Tests" << std::endl;
    std::cout << "12. GPU Info" << std::endl;
    std::cout << "0.  Exit" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Choose an option: ";
}

// Performans özeti
struct PerformanceStats {
    std::string algorithm;
    float custom_time_ms;
    float thrust_time_ms;
    float cub_time_ms;
    float speedup_vs_thrust;
    float speedup_vs_cub;
    
    PerformanceStats(const std::string& alg) : algorithm(alg), 
        custom_time_ms(0), thrust_time_ms(0), cub_time_ms(0),
        speedup_vs_thrust(0), speedup_vs_cub(0) {}
};

std::vector<PerformanceStats> performance_results;

// Performans sonuçlarını yazdır
void print_performance_summary() {
    if (performance_results.empty()) {
        std::cout << "No performance data available." << std::endl;
        return;
    }
    
    std::cout << "\n=== PERFORMANCE SUMMARY ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Algorithm" 
              << std::setw(12) << "Custom(ms)" 
              << std::setw(12) << "Thrust(ms)" 
              << std::setw(12) << "CUB(ms)"
              << std::setw(12) << "vs Thrust"
              << std::setw(12) << "vs CUB" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& stat : performance_results) {
        std::cout << std::left << std::setw(20) << stat.algorithm
                  << std::setw(12) << std::fixed << std::setprecision(2) << stat.custom_time_ms
                  << std::setw(12) << stat.thrust_time_ms
                  << std::setw(12) << stat.cub_time_ms
                  << std::setw(12) << std::setprecision(1) << stat.speedup_vs_thrust << "x"
                  << std::setw(12) << stat.speedup_vs_cub << "x" << std::endl;
    }
    std::cout << "============================\n" << std::endl;
}

// CUDA Graphs demonstrasyonu
void demonstrate_cuda_graphs() {
    std::cout << "\n=== CUDA GRAPHS DEMONSTRATION ===" << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 100);
    
    ManagedMemory<int> d_input(n);
    ManagedMemory<int> d_output1(n);
    ManagedMemory<int> d_output2(n);
    ManagedMemory<int> d_final_output(1);
    
    d_input.copy_from_host(input.data());
    
    // Regular kernel launches
    CudaTimer timer_regular;
    timer_regular.start();
    
    for (int i = 0; i < 10; ++i) {
        // Prefix sum -> Reduce -> Copy sequence
        thrust::device_ptr<int> thrust_input(d_input.get());
        thrust::device_ptr<int> thrust_output1(d_output1.get());
        thrust::device_ptr<int> thrust_output2(d_output2.get());
        thrust::device_ptr<int> thrust_final(d_final_output.get());
        
        thrust::inclusive_scan(thrust_input, thrust_input + n, thrust_output1);
        *thrust_final = thrust::reduce(thrust_output1, thrust_output1 + n);
        cudaDeviceSynchronize();
    }
    
    timer_regular.stop();
    std::cout << "Regular launches (10x): " << timer_regular.elapsed_ms() << " ms" << std::endl;
    
    // CUDA Graphs
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CudaTimer timer_graph;
    timer_graph.start();
    
    // Graph oluştur
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Graph içinde kernel sequence
    thrust::device_ptr<int> thrust_input(d_input.get());
    thrust::device_ptr<int> thrust_output1(d_output1.get());
    thrust::device_ptr<int> thrust_final(d_final_output.get());
    
    thrust::inclusive_scan(thrust::cuda::par.on(stream), thrust_input, thrust_input + n, thrust_output1);
    *thrust_final = thrust::reduce(thrust::cuda::par.on(stream), thrust_output1, thrust_output1 + n);
    
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    
    // Graph'ı 10 kez çalıştır
    for (int i = 0; i < 10; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    timer_graph.stop();
    std::cout << "CUDA Graphs (10x): " << timer_graph.elapsed_ms() << " ms" << std::endl;
    
    float speedup = timer_regular.elapsed_ms() / timer_graph.elapsed_ms();
    std::cout << "CUDA Graphs Speedup: " << std::fixed << std::setprecision(2) 
              << speedup << "x" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// Memory bandwidth test
void test_memory_bandwidth() {
    std::cout << "\n=== MEMORY BANDWIDTH TEST ===" << std::endl;
    
    const size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    std::cout << std::left << std::setw(12) << "Size(KB)" 
              << std::setw(15) << "H2D(GB/s)" 
              << std::setw(15) << "D2H(GB/s)"
              << std::setw(15) << "D2D(GB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];
        size_t bytes = size * sizeof(float);
        
        // Host memory
        std::vector<float> h_data(size, 1.0f);
        
        // Device memory
        float* d_data1;
        float* d_data2;
        CUDA_CHECK(cudaMalloc(&d_data1, bytes));
        CUDA_CHECK(cudaMalloc(&d_data2, bytes));
        
        // Host to Device
        CudaTimer timer;
        const int iterations = 100;
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaMemcpy(d_data1, h_data.data(), bytes, cudaMemcpyHostToDevice));
        }
        timer.stop();
        float h2d_time = timer.elapsed_ms() / iterations;
        float h2d_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (h2d_time / 1000.0f);
        
        // Device to Host
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
        // Device to Host
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaMemcpy(h_data.data(), d_data1, bytes, cudaMemcpyDeviceToHost));
        }
        timer.stop();
        float d2h_time = timer.elapsed_ms() / iterations;
        float d2h_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2h_time / 1000.0f);
        
        // Device to Device
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaMemcpy(d_data2, d_data1, bytes, cudaMemcpyDeviceToDevice));
        }
        timer.stop();
        float d2d_time = timer.elapsed_ms() / iterations;
        float d2d_bandwidth = (bytes / (1024.0f * 1024.0f * 1024.0f)) / (d2d_time / 1000.0f);
        
        std::cout << std::left << std::setw(12) << size/1024 
                  << std::setw(15) << std::fixed << std::setprecision(1) << h2d_bandwidth
                  << std::setw(15) << d2h_bandwidth
                  << std::setw(15) << d2d_bandwidth << std::endl;
        
        CUDA_CHECK(cudaFree(d_data1));
        CUDA_CHECK(cudaFree(d_data2));
    }
}

// Occupancy analysis
template<typename KernelFunc>
void analyze_kernel_occupancy(KernelFunc kernel, const std::string& kernel_name) {
    std::cout << "\n=== OCCUPANCY ANALYSIS: " << kernel_name << " ===" << std::endl;
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Test different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    std::cout << std::left << std::setw(12) << "Block Size" 
              << std::setw(15) << "Occupancy(%)" 
              << std::setw(15) << "Active Blocks"
              << std::setw(15) << "Active Threads" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int i = 0; i < num_block_sizes; ++i) {
        int block_size = block_sizes[i];
        if (block_size > prop.maxThreadsPerBlock) continue;
        
        int max_active_blocks;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, kernel, block_size, 0));
        
        float occupancy = (max_active_blocks * block_size) / 
                         (float)prop.maxThreadsPerMultiProcessor * 100.0f;
        
        int total_active_blocks = max_active_blocks * prop.multiProcessorCount;
        int total_active_threads = total_active_blocks * block_size;
        
        std::cout << std::left << std::setw(12) << block_size
                  << std::setw(15) << std::fixed << std::setprecision(1) << occupancy
                  << std::setw(15) << total_active_blocks
                  << std::setw(15) << total_active_threads << std::endl;
    }
    
    // Optimal block size
    int min_grid_size, block_size_opt;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_opt, kernel, 0, 0));
    
    std::cout << "\nOptimal block size: " << block_size_opt << std::endl;
    std::cout << "Minimum grid size for max occupancy: " << min_grid_size << std::endl;
}

// Ana program
int main() {
    // CUDA initialization
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::cout << "CUDA Parallel Algorithms Collection" << std::endl;
    std::cout << "RTX 4070 Ti Super | CUDA 12.4" << std::endl;
    
    print_gpu_info();
    
    int choice;
    bool running = true;
    
    while (running) {
        show_menu();
        std::cin >> choice;
        
        switch (choice) {
            case 1:
                test_prefix_sum();
                break;
            case 2:
                test_reduce();
                break;
            case 3:
                test_histogram();
                break;
            case 4:
                test_radix_sort();
                break;
            case 5:
                test_bfs();
                break;
            case 6:
                test_scan();
                break;
            case 7:
                test_compact();
                break;
            case 8:
                test_matrix_multiply();
                break;
            case 9:
                test_merge_sort();
                break;
            case 10:
                test_convolution();
                break;
            case 11:
                std::cout << "\n=== RUNNING ALL TESTS ===" << std::endl;
                test_prefix_sum();
                test_reduce();
                test_histogram();
                test_radix_sort();
                test_bfs();
                test_scan();
                test_compact();
                test_matrix_multiply();
                test_merge_sort();
                test_convolution();
                print_performance_summary();
                demonstrate_cuda_graphs();
                test_memory_bandwidth();
                break;
            case 12:
                print_gpu_info();
                demonstrate_cuda_graphs();
                test_memory_bandwidth();
                break;
            case 0:
                running = false;
                break;
            default:
                std::cout << "Invalid option! Please try again." << std::endl;
                break;
        }
        
        if (choice != 0 && choice != 12) {
            std::cout << "\nPress Enter to continue...";
            std::cin.ignore();
            std::cin.get();
        }
    }
    
    std::cout << "\nThank you for using CUDA Parallel Algorithms Collection!" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}

// Stub implementations for remaining algorithms
void test_scan() {
    std::cout << "\n=== SCAN ALGORITHM TEST ===" << std::endl;
    std::cout << "Advanced scan algorithms (inclusive/exclusive, segmented scan) will be implemented here." << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 10);
    
    // Thrust exclusive scan
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_output(n);
    
    CudaTimer timer;
    timer.start();
    thrust::exclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    timer.stop();
    
    std::cout << "Thrust Exclusive Scan - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Verify first 10 elements
    thrust::host_vector<int> h_output = d_output;
    std::cout << "First 10 elements - Input: ";
    for (int i = 0; i < 10; ++i) std::cout << input[i] << " ";
    std::cout << std::endl;
    std::cout << "First 10 elements - Exclusive Scan: ";
    for (int i = 0; i < 10; ++i) std::cout << h_output[i] << " ";
    std::cout << std::endl;
}

void test_compact() {
    std::cout << "\n=== STREAM COMPACTION TEST ===" << std::endl;
    std::cout << "Stream compaction (removing elements based on predicate) will be implemented here." << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<int>(n, 1, 100);
    
    // Thrust copy_if (stream compaction)
    thrust::device_vector<int> d_input(input);
    thrust::device_vector<int> d_output(n);
    
    CudaTimer timer;
    timer.start();
    auto end = thrust::copy_if(d_input.begin(), d_input.end(), d_output.begin(),
                              [] __device__ (int x) { return x % 2 == 0; }); // Even numbers only
    timer.stop();
    
    int compacted_size = end - d_output.begin();
    std::cout << "Thrust Stream Compaction - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    std::cout << "Original size: " << n << ", Compacted size: " << compacted_size << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
              << (float)compacted_size / n * 100 << "%" << std::endl;
}

void test_matrix_multiply() {
    std::cout << "\n=== MATRIX MULTIPLICATION TEST ===" << std::endl;
    std::cout << "Optimized matrix multiplication (tiled, shared memory) will be implemented here." << std::endl;
    
    const int size = 1024;
    auto matrix_a = generate_random_data<float>(size * size, 0.0f, 1.0f);
    auto matrix_b = generate_random_data<float>(size * size, 0.0f, 1.0f);
    
    // cuBLAS referans
    thrust::device_vector<float> d_a(matrix_a);
    thrust::device_vector<float> d_b(matrix_b);
    thrust::device_vector<float> d_c(size * size);
    
    CudaTimer timer;
    timer.start();
    
    // Simple matrix multiplication using Thrust
    // This is just a placeholder - real implementation would use cuBLAS or custom kernels
    thrust::fill(d_c.begin(), d_c.end(), 0.0f);
    
    timer.stop();
    
    std::cout << "Matrix Multiplication (" << size << "x" << size << ") - Time: " 
              << timer.elapsed_ms() << " ms" << std::endl;
    
    // GFLOPS calculation
    double operations = 2.0 * size * size * size; // 2*N^3 operations
    double gflops = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    std::cout << "Performance: " << std::fixed << std::setprecision(2) 
              << gflops << " GFLOPS" << std::endl;
}

void test_merge_sort() {
    std::cout << "\n=== MERGE SORT TEST ===" << std::endl;
    std::cout << "GPU merge sort implementation will be implemented here." << std::endl;
    
    const int n = 1000000;
    auto input = generate_random_data<unsigned int>(n, 0, 0xFFFFFFFF);
    
    // Thrust sort as reference
    thrust::device_vector<unsigned int> d_input(input);
    
    CudaTimer timer;
    timer.start();
    thrust::sort(d_input.begin(), d_input.end());
    timer.stop();
    
    std::cout << "Thrust Sort - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Verify sorting
    thrust::host_vector<unsigned int> h_output = d_input;
    bool is_sorted = std::is_sorted(h_output.begin(), h_output.end());
    std::cout << "Result is sorted: " << (is_sorted ? "✓" : "✗") << std::endl;
}

void test_convolution() {
    std::cout << "\n=== CONVOLUTION TEST ===" << std::endl;
    std::cout << "1D/2D convolution with shared memory optimization will be implemented here." << std::endl;
    
    const int signal_size = 1000000;
    const int kernel_size = 128;
    
    auto signal = generate_random_data<float>(signal_size, -1.0f, 1.0f);
    auto kernel = generate_random_data<float>(kernel_size, -0.1f, 0.1f);
    
    // Simple convolution using Thrust
    thrust::device_vector<float> d_signal(signal);
    thrust::device_vector<float> d_kernel(kernel);
    thrust::device_vector<float> d_output(signal_size);
    
    CudaTimer timer;
    timer.start();
    
    // Placeholder implementation
    thrust::fill(d_output.begin(), d_output.end(), 0.0f);
    
    timer.stop();
    
    std::cout << "1D Convolution (signal: " << signal_size << ", kernel: " << kernel_size 
              << ") - Time: " << timer.elapsed_ms() << " ms" << std::endl;
    
    // Throughput calculation
    double operations = (double)signal_size * kernel_size;
    double throughput = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " GOPS" << std::endl;
}