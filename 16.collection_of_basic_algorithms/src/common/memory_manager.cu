// ==================== src/common/memory_manager.cu ====================
#include "../../include/common.h"

// Performance metrics implementation
void PerformanceMetrics::print_summary(const std::string& algorithm_name) const {
    std::cout << "\n=== " << algorithm_name << " Performance Summary ===" << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(3) 
              << execution_time_ms << " ms" << std::endl;
    
    if (bandwidth_gb_s > 0) {
        std::cout << "Memory Bandwidth: " << std::setprecision(2) 
                  << bandwidth_gb_s << " GB/s" << std::endl;
    }
    
    if (throughput_gops > 0) {
        std::cout << "Throughput: " << std::setprecision(2) 
                  << throughput_gops << " GOPS" << std::endl;
    }
    
    if (occupancy_percent > 0) {
        std::cout << "Occupancy: " << std::setprecision(1) 
                  << occupancy_percent << "%" << std::endl;
    }
    
    if (shared_memory_bytes > 0) {
        std::cout << "Shared Memory: " << shared_memory_bytes / 1024 << " KB" << std::endl;
    }
    
    std::cout << "=================================" << std::endl;
}

// Bandwidth measurement
BandwidthMeasurer::BandwidthResult BandwidthMeasurer::measure_memory_bandwidth(
    size_t size_bytes, int iterations) {
    
    BandwidthResult result;
    result.data_size_bytes = size_bytes;
    
    std::vector<char> h_data(size_bytes);
    char* d_data1;
    char* d_data2;
    
    CUDA_CHECK(cudaMalloc(&d_data1, size_bytes));
    CUDA_CHECK(cudaMalloc(&d_data2, size_bytes));
    
    CudaTimer timer;
    
    // Host to Device
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_data1, h_data.data(), size_bytes, cudaMemcpyHostToDevice));
    }
    timer.stop();
    result.h2d_time_ms = timer.elapsed_ms() / iterations;
    result.h2d_bandwidth_gb_s = (size_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                               (result.h2d_time_ms / 1000.0);
    
    // Device to Host
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(h_data.data(), d_data1, size_bytes, cudaMemcpyDeviceToHost));
    }
    timer.stop();
    result.d2h_time_ms = timer.elapsed_ms() / iterations;
    result.d2h_bandwidth_gb_s = (size_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                               (result.d2h_time_ms / 1000.0);
    
    // Device to Device
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaMemcpy(d_data2, d_data1, size_bytes, cudaMemcpyDeviceToDevice));
    }
    timer.stop();
    result.d2d_time_ms = timer.elapsed_ms() / iterations;
    result.d2d_bandwidth_gb_s = (size_bytes / (1024.0 * 1024.0 * 1024.0)) / 
                               (result.d2d_time_ms / 1000.0);
    
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
    
    return result;
}

void BandwidthMeasurer::print_bandwidth_table(const std::vector<size_t>& sizes) {
    std::cout << "\n=== MEMORY BANDWIDTH ANALYSIS ===" << std::endl;
    std::cout << std::left << std::setw(12) << "Size(MB)" 
              << std::setw(12) << "H2D(GB/s)" 
              << std::setw(12) << "D2H(GB/s)"
              << std::setw(12) << "D2D(GB/s)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (size_t size : sizes) {
        auto result = measure_memory_bandwidth(size * 1024 * 1024); // Convert MB to bytes
        
        std::cout << std::left << std::setw(12) << size
                  << std::setw(12) << std::fixed << std::setprecision(1) 
                  << result.h2d_bandwidth_gb_s
                  << std::setw(12) << result.d2h_bandwidth_gb_s
                  << std::setw(12) << result.d2d_bandwidth_gb_s << std::endl;
    }
}

double BandwidthMeasurer::get_theoretical_bandwidth() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // RTX 4070 Ti Super: 504 GB/s theoretical bandwidth
    // Calculate from memory clock and bus width
    double memory_clock_hz = prop.memoryClockRate * 1000.0; // Convert to Hz
    double bus_width_bytes = prop.memoryBusWidth / 8.0;     // Convert bits to bytes
    
    // GDDR6X uses DDR, so effective rate is 2x
    return 2.0 * memory_clock_hz * bus_width_bytes / 1e9; // Convert to GB/s
}

// Performance statistics
PerformanceStats::PerformanceStats(const std::string& alg) : algorithm(alg) {
    custom_time_ms = 0;
    thrust_time_ms = 0; 
    cub_time_ms = 0;
    speedup_vs_thrust = 0;
    speedup_vs_cub = 0;
}

std::vector<PerformanceStats> performance_results;

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