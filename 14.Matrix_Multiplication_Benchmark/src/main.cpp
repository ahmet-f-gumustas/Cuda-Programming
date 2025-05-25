#include "matrix_benchmark.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "=== CUDA Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << std::endl;
    
    // Sistem bilgilerini yazdır
    print_system_info();
    std::cout << std::endl;
    
    // Test edilecek matrix boyutları
    std::vector<int> matrix_sizes = {128, 256, 512, 1024};
    
    // Komut satırından boyut belirtilmişse kullan
    if (argc > 1) {
        matrix_sizes.clear();
        for (int i = 1; i < argc; i++) {
            int size = std::atoi(argv[i]);
            if (size > 0 && size <= 4096) {
                matrix_sizes.push_back(size);
            } else {
                std::cerr << "Geçersiz matrix boyutu: " << argv[i] << std::endl;
                std::cerr << "Boyut 1-4096 arasında olmalıdır." << std::endl;
                return 1;
            }
        }
    }
    
    std::cout << "Test edilecek matrix boyutları: ";
    for (size_t i = 0; i < matrix_sizes.size(); i++) {
        std::cout << matrix_sizes[i];
        if (i < matrix_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;
    
    // Her boyut için benchmark çalıştır
    std::vector<BenchmarkResult> all_results;
    
    for (int size : matrix_sizes) {
        std::cout << "Matrix boyutu " << size << "x" << size << " test ediliyor..." << std::endl;
        
        try {
            BenchmarkResult result = run_benchmark(size);
            all_results.push_back(result);
            print_benchmark_results(result);
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Hata oluştu (boyut " << size << "): " << e.what() << std::endl;
            continue;
        }
    }
    
    // Özet sonuçlar
    if (!all_results.empty()) {
        std::cout << "=== ÖZET SONUÇLAR ===" << std::endl;
        std::cout << std::left << std::setw(8) << "Boyut" 
                  << std::setw(12) << "CPU (ms)" 
                  << std::setw(12) << "GPU Naive" 
                  << std::setw(12) << "GPU Tiled" 
                  << std::setw(12) << "cuBLAS" 
                  << std::setw(10) << "Hızlanma" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        for (const auto& result : all_results) {
            double speedup = result.cpu_time_ms / result.gpu_cublas_time_ms;
            std::cout << std::left << std::setw(8) << result.matrix_size
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.cpu_time_ms
                      << std::setw(12) << result.gpu_naive_time_ms
                      << std::setw(12) << result.gpu_tiled_time_ms
                      << std::setw(12) << result.gpu_cublas_time_ms
                      << std::setw(10) << speedup << "x" << std::endl;
        }
    }
    
    std::cout << std::endl << "Benchmark tamamlandı!" << std::endl;
    return 0;
}