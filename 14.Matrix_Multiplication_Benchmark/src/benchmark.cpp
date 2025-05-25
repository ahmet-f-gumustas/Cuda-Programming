#include "matrix_benchmark.h"
#include <iostream>
#include <iomanip>
#include <chrono>

void cpu_matrix_multiply(const Matrix& A, const Matrix& B, Matrix& C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

BenchmarkResult run_benchmark(int N) {
    BenchmarkResult result = {};
    result.matrix_size = N;
    result.operations = static_cast<size_t>(N) * N * N * 2; // Her element için N çarpım ve N-1 toplama
    
    size_t matrix_bytes = N * N * sizeof(float);
    
    // CPU matrislerini hazırla
    Matrix A(N * N), B(N * N), C_cpu(N * N), C_gpu(N * N);
    
    std::cout << "  Matrisler başlatılıyor..." << std::endl;
    initialize_matrix(A, N, true);
    initialize_matrix(B, N, true);
    
    // GPU bellek alanlarını ayır
    std::cout << "  GPU belleği ayrılıyor..." << std::endl;
    auto d_A = allocate_device_memory(matrix_bytes);
    auto d_B = allocate_device_memory(matrix_bytes);
    auto d_C = allocate_device_memory(matrix_bytes);
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK(cudaMemcpy(d_A.get(), A.data(), matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.get(), B.data(), matrix_bytes, cudaMemcpyHostToDevice));
    
    // cuBLAS handle oluştur
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // 1. CPU Benchmark
    std::cout << "  CPU hesaplama çalıştırılıyor..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cpu_matrix_multiply(A, B, C_cpu, N);
    auto end = std::chrono::high_resolution_clock::now();
    result.cpu_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 2. GPU Naive Benchmark
    std::cout << "  GPU Naive hesaplama çalıştırılıyor..." << std::endl;
    CUDA_CHECK(cudaMemset(d_C.get(), 0, matrix_bytes));
    
    start = std::chrono::high_resolution_clock::now();
    gpu_matrix_multiply_naive(d_A.get(), d_B.get(), d_C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    result.gpu_naive_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Sonucu kontrol et
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C.get(), matrix_bytes, cudaMemcpyDeviceToHost));
    bool naive_correct = verify_results(C_cpu, C_gpu);
    
    // 3. GPU Tiled Benchmark
    std::cout << "  GPU Tiled hesaplama çalıştırılıyor..." << std::endl;
    CUDA_CHECK(cudaMemset(d_C.get(), 0, matrix_bytes));
    
    start = std::chrono::high_resolution_clock::now();
    gpu_matrix_multiply_tiled(d_A.get(), d_B.get(), d_C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    result.gpu_tiled_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Sonucu kontrol et
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C.get(), matrix_bytes, cudaMemcpyDeviceToHost));
    bool tiled_correct = verify_results(C_cpu, C_gpu);
    
    // 4. cuBLAS Benchmark
    std::cout << "  cuBLAS hesaplama çalıştırılıyor..." << std::endl;
    CUDA_CHECK(cudaMemset(d_C.get(), 0, matrix_bytes));
    
    start = std::chrono::high_resolution_clock::now();
    cublas_matrix_multiply(cublas_handle, d_A.get(), d_B.get(), d_C.get(), N);
    end = std::chrono::high_resolution_clock::now();
    result.gpu_cublas_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Sonucu kontrol et
    CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C.get(), matrix_bytes, cudaMemcpyDeviceToHost));
    bool cublas_correct = verify_results(C_cpu, C_gpu, 1e-2); // cuBLAS için daha gevşek tolerans
    
    result.results_match = naive_correct && tiled_correct && cublas_correct;
    result.max_error = calculate_max_error(C_cpu, C_gpu);
    
    // Temizlik
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
    return result;
}

void print_benchmark_results(const BenchmarkResult& result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Sonuçlar:" << std::endl;
    std::cout << "    CPU Zamanı:      " << std::setw(8) << result.cpu_time_ms << " ms" << std::endl;
    std::cout << "    GPU Naive:       " << std::setw(8) << result.gpu_naive_time_ms << " ms "
              << "(Hızlanma: " << result.cpu_time_ms / result.gpu_naive_time_ms << "x)" << std::endl;
    std::cout << "    GPU Tiled:       " << std::setw(8) << result.gpu_tiled_time_ms << " ms "
              << "(Hızlanma: " << result.cpu_time_ms / result.gpu_tiled_time_ms << "x)" << std::endl;
    std::cout << "    cuBLAS:          " << std::setw(8) << result.gpu_cublas_time_ms << " ms "
              << "(Hızlanma: " << result.cpu_time_ms / result.gpu_cublas_time_ms << "x)" << std::endl;
    
    // Performans metrikleri
    double gflops_cpu = (result.operations / 1e9) / (result.cpu_time_ms / 1000.0);
    double gflops_cublas = (result.operations / 1e9) / (result.gpu_cublas_time_ms / 1000.0);
    
    std::cout << "    CPU GFLOPS:      " << std::setw(8) << gflops_cpu << std::endl;
    std::cout << "    cuBLAS GFLOPS:   " << std::setw(8) << gflops_cublas << std::endl;
    
    // Doğruluk kontrolü
    std::cout << "    Sonuçlar Doğru:  " << (result.results_match ? "✓" : "✗") << std::endl;
    std::cout << "    Maksimum Hata:   " << std::scientific << std::setprecision(2) 
              << result.max_error << std::endl;
}