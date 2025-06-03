#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <functional>

// High-precision CUDA timer
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    bool started = false;
    
public:
    CudaTimer();
    ~CudaTimer();
    
    // No copy, only move
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    
    CudaTimer(CudaTimer&& other) noexcept;
    CudaTimer& operator=(CudaTimer&& other) noexcept;
    
    void start();
    void stop();
    float elapsed_ms();
    void reset();
};

// CPU timer for comparison
class CpuTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool started = false;
    
public:
    void start();
    void stop();
    float elapsed_ms();
    double elapsed_us();
};

// Performance metrics collection
struct PerformanceMetrics {
    float execution_time_ms = 0.0f;
    double bandwidth_gb_s = 0.0;
    double throughput_gops = 0.0;
    float occupancy_percent = 0.0f;
    size_t shared_memory_bytes = 0;
    size_t register_count = 0;
    size_t bytes_transferred = 0;
    
    void print_summary(const std::string& algorithm_name) const;
    void save_to_json(const std::string& filename) const;
};

// Memory bandwidth measurement
class BandwidthMeasurer {
public:
    struct BandwidthResult {
        double h2d_bandwidth_gb_s;
        double d2h_bandwidth_gb_s;
        double d2d_bandwidth_gb_s;
        size_t data_size_bytes;
        float h2d_time_ms;
        float d2h_time_ms;
        float d2d_time_ms;
    };
    
    static BandwidthResult measure_memory_bandwidth(size_t size_bytes, int iterations = 100);
    static void print_bandwidth_table(const std::vector<size_t>& sizes);
    static double get_theoretical_bandwidth();
};

// Occupancy analyzer
class OccupancyAnalyzer {
public:
    struct OccupancyResult {
        float theoretical_occupancy;
        float achieved_occupancy;
        int optimal_block_size;
        int max_active_blocks;
        int registers_per_thread;
        size_t shared_memory_per_block;
    };
    
    template<typename KernelFunc>
    static OccupancyResult analyze_kernel(KernelFunc kernel, int block_size, 
                                         size_t shared_mem_size = 0);
    
    template<typename KernelFunc>
    static void print_occupancy_analysis(KernelFunc kernel, const std::string& kernel_name);
    
    template<typename KernelFunc>
    static int find_optimal_block_size(KernelFunc kernel);
};

// Performance profiler
class PerformanceProfiler {
private:
    std::string current_algorithm;
    std::vector<PerformanceMetrics> measurements;
    
public:
    void start_algorithm(const std::string& name);
    void end_algorithm();
    
    template<typename Func>
    PerformanceMetrics measure_function(Func&& func, size_t bytes_processed = 0);
    
    void add_measurement(const PerformanceMetrics& metrics);
    void print_summary() const;
    void save_results(const std::string& filename) const;
    void clear();
    
    const std::vector<PerformanceMetrics>& get_measurements() const { return measurements; }
};

// Benchmark runner
class BenchmarkRunner {
public:
    struct BenchmarkConfig {
        std::vector<size_t> data_sizes = {1000, 10000, 100000, 1000000, 10000000};
        int iterations = 10;
        int warmup_iterations = 3;
        bool measure_bandwidth = true;
        bool measure_occupancy = true;
        bool save_results = true;
        std::string output_directory = "benchmark_results";
    };
    
private:
    BenchmarkConfig config;
    PerformanceProfiler profiler;
    
public:
    explicit BenchmarkRunner(const BenchmarkConfig& cfg = BenchmarkConfig{});
    
    template<typename AlgorithmFunc>
    void run_scaling_benchmark(const std::string& algorithm_name, AlgorithmFunc&& algorithm);
    
    template<typename CustomFunc, typename ThrustFunc, typename CubFunc>
    void run_comparison_benchmark(const std::string& algorithm_name,
                                 CustomFunc&& custom_impl,
                                 ThrustFunc&& thrust_impl,
                                 CubFunc&& cub_impl);
    
    void run_memory_bandwidth_test();
    void print_summary() const;
    void save_results() const;
};

// CUDA error checking with performance impact measurement
class CudaErrorChecker {
public:
    static void check_last_error(const char* file, int line);
    static void check_cuda_call(cudaError_t error, const char* file, int line);
    
    // Performance-aware error checking (can be disabled in release builds)
    static void check_kernel_performance(const char* kernel_name, 
                                       float expected_min_time_ms = 0.0f);
};

// Utility macros for performance measurement
#define CUDA_CHECK_PERF(call) do { \
    cudaError_t err = call; \
    CudaErrorChecker::check_cuda_call(err, __FILE__, __LINE__); \
} while(0)

#define CUDA_CHECK_KERNEL_PERF() do { \
    CudaErrorChecker::check_last_error(__FILE__, __LINE__); \
    cudaDeviceSynchronize(); \
} while(0)

#define MEASURE_FUNCTION(profiler, func, bytes) \
    profiler.measure_function([&](){ func; }, bytes)

// Template implementations (must be in header for template instantiation)

template<typename KernelFunc>
OccupancyAnalyzer::OccupancyResult OccupancyAnalyzer::analyze_kernel(
    KernelFunc kernel, int block_size, size_t shared_mem_size) {
    
    OccupancyResult result;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate theoretical occupancy
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, 
                                                 block_size, shared_mem_size);
    
    result.theoretical_occupancy = (max_active_blocks * block_size) / 
                                  (float)prop.maxThreadsPerMultiProcessor * 100.0f;
    
    // Find optimal block size
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &result.optimal_block_size, 
                                      kernel, shared_mem_size, 0);
    
    result.max_active_blocks = max_active_blocks * prop.multiProcessorCount;
    result.shared_memory_per_block = shared_mem_size;
    
    // Note: Register count and achieved occupancy would need runtime measurement
    result.registers_per_thread = 0; // Requires compilation analysis
    result.achieved_occupancy = 0.0f; // Requires profiler integration
    
    return result;
}

template<typename KernelFunc>
void OccupancyAnalyzer::print_occupancy_analysis(KernelFunc kernel, 
                                                const std::string& kernel_name) {
    std::cout << "\n=== OCCUPANCY ANALYSIS: " << kernel_name << " ===" << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Test different block sizes
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    
    std::cout << std::left << std::setw(12) << "Block Size" 
              << std::setw(15) << "Occupancy(%)" 
              << std::setw(15) << "Active Blocks"
              << std::setw(15) << "Active Threads" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int block_size : block_sizes) {
        if (block_size > prop.maxThreadsPerBlock) continue;
        
        auto result = analyze_kernel(kernel, block_size);
        
        std::cout << std::left << std::setw(12) << block_size
                  << std::setw(15) << std::fixed << std::setprecision(1) 
                  << result.theoretical_occupancy
                  << std::setw(15) << result.max_active_blocks
                  << std::setw(15) << result.max_active_blocks * block_size << std::endl;
    }
    
    auto optimal_result = analyze_kernel(kernel, 0); // Use optimal block size
    std::cout << "\nOptimal block size: " << optimal_result.optimal_block_size << std::endl;
}

template<typename KernelFunc>
int OccupancyAnalyzer::find_optimal_block_size(KernelFunc kernel) {
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);
    return block_size;
}

template<typename Func>
PerformanceMetrics PerformanceProfiler::measure_function(Func&& func, size_t bytes_processed) {
    PerformanceMetrics metrics;
    
    CudaTimer timer;
    timer.start();
    
    func();
    
    timer.stop();
    metrics.execution_time_ms = timer.elapsed_ms();
    metrics.bytes_transferred = bytes_processed;
    
    if (bytes_processed > 0) {
        metrics.bandwidth_gb_s = bytes_processed / (metrics.execution_time_ms / 1000.0) / 1e9;
    }
    
    return metrics;
}

template<typename AlgorithmFunc>
void BenchmarkRunner::run_scaling_benchmark(const std::string& algorithm_name, 
                                           AlgorithmFunc&& algorithm) {
    std::cout << "\n=== SCALING BENCHMARK: " << algorithm_name << " ===" << std::endl;
    
    profiler.start_algorithm(algorithm_name);
    
    for (size_t data_size : config.data_sizes) {
        std::vector<float> times;
        
        // Warmup
        for (int i = 0; i < config.warmup_iterations; ++i) {
            algorithm(data_size);
        }
        
        // Actual measurements
        for (int i = 0; i < config.iterations; ++i) {
            auto metrics = profiler.measure_function([&](){ algorithm(data_size); }, 
                                                   data_size * sizeof(int));
            times.push_back(metrics.execution_time_ms);
        }
        
        // Calculate statistics
        float min_time = *std::min_element(times.begin(), times.end());
        float max_time = *std::max_element(times.begin(), times.end());
        float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        
        double throughput = data_size / (avg_time / 1000.0) / 1e9; // GOPS
        
        std::cout << "Size: " << std::setw(8) << data_size 
                  << " | Time: " << std::setw(8) << std::fixed << std::setprecision(3) << avg_time << " ms"
                  << " | Throughput: " << std::setw(8) << std::setprecision(2) << throughput << " GOPS"
                  << std::endl;
    }
    
    profiler.end_algorithm();
}

template<typename CustomFunc, typename ThrustFunc, typename CubFunc>
void BenchmarkRunner::run_comparison_benchmark(const std::string& algorithm_name,
                                              CustomFunc&& custom_impl,
                                              ThrustFunc&& thrust_impl,
                                              CubFunc&& cub_impl) {
    std::cout << "\n=== COMPARISON BENCHMARK: " << algorithm_name << " ===" << std::endl;
    
    const size_t test_size = 1000000; // 1M elements for comparison
    
    // Custom implementation
    auto custom_metrics = profiler.measure_function([&](){ custom_impl(test_size); });
    
    // Thrust implementation  
    auto thrust_metrics = profiler.measure_function([&](){ thrust_impl(test_size); });
    
    // CUB implementation
    auto cub_metrics = profiler.measure_function([&](){ cub_impl(test_size); });
    
    // Print comparison
    std::cout << std::left << std::setw(15) << "Implementation" 
              << std::setw(12) << "Time(ms)" 
              << std::setw(15) << "Speedup vs Custom" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    std::cout << std::left << std::setw(15) << "Custom"
              << std::setw(12) << std::fixed << std::setprecision(3) << custom_metrics.execution_time_ms
              << std::setw(15) << "1.0x" << std::endl;
              
    std::cout << std::left << std::setw(15) << "Thrust"
              << std::setw(12) << thrust_metrics.execution_time_ms
              << std::setw(15) << std::setprecision(2) 
              << custom_metrics.execution_time_ms / thrust_metrics.execution_time_ms << "x" << std::endl;
              
    std::cout << std::left << std::setw(15) << "CUB"
              << std::setw(12) << cub_metrics.execution_time_ms  
              << std::setw(15) << custom_metrics.execution_time_ms / cub_metrics.execution_time_ms << "x" << std::endl;
}