// ==================== src/common/cuda_utils.cu ====================
#include "../../include/common.h"

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

// CUDA Graphs demonstration
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