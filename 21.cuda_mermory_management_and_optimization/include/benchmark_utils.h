#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::string name;

public:
    BenchmarkTimer(const std::string& benchmark_name) : name(benchmark_name) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~BenchmarkTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << std::setw(30) << std::left << name 
                  << ": " << std::setw(10) << std::right << duration.count() 
                  << " µs" << std::endl;
    }
};

// Performans metrikleri
struct PerformanceMetrics {
    float bandwidth_gb_s;
    float compute_gflops;
    float latency_ms;
    
    void print() const {
        std::cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "Compute: " << compute_gflops << " GFLOPS" << std::endl;
        std::cout << "Latency: " << latency_ms << " ms" << std::endl;
    }
};

#endif // BENCHMARK_UTILS_H