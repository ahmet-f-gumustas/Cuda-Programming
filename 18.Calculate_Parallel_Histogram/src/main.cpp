#include <iostream>
#include <vector>
#include <iomanip>

#include "histogram_cpu.h"
#include "histogram_cuda.h"
#include "utils.h"

int main() {
    std::cout << "=== CUDA Paralel Histogram Hesaplama Projesi ===" << std::endl;
    std::cout << std::endl;
    
    // GPU bilgilerini yazdır
    Utils::printGPUInfo();
    std::cout << std::endl;
    
    // Test parametreleri
    const int DATA_SIZE = 1000000;  // 1 milyon veri noktası
    const int NUM_BINS = 256;       // 256 bin (0-255 değer aralığı için)
    const int MIN_VALUE = 0;
    const int MAX_VALUE = 255;
    
    std::cout << "Test Parametreleri:" << std::endl;
    std::cout << "- Veri boyutu: " << DATA_SIZE << std::endl;
    std::cout << "- Bin sayısı: " << NUM_BINS << std::endl;
    std::cout << "- Değer aralığı: [" << MIN_VALUE << ", " << MAX_VALUE << "]" << std::endl;
    std::cout << std::endl;
    
    // Test verilerini oluştur
    std::cout << "Test verileri oluşturuluyor..." << std::endl;
    auto random_data = Utils::generateRandomData(DATA_SIZE, MIN_VALUE, MAX_VALUE);
    auto gaussian_data = Utils::generateGaussianData(DATA_SIZE, 128.0, 40.0);
    
    std::cout << "Random veri istatistikleri:" << std::endl;
    Utils::printDataStats(random_data);
    std::cout << std::endl;
    
    std::cout << "Gaussian veri istatistikleri:" << std::endl;
    Utils::printDataStats(gaussian_data);
    std::cout << std::endl;
    
    // Histogram sonuçları için vektörler
    std::vector<int> hist_cpu_single(NUM_BINS, 0);
    std::vector<int> hist_cpu_parallel(NUM_BINS, 0);
    std::vector<int> hist_cuda_basic(NUM_BINS, 0);
    std::vector<int> hist_cuda_optimized(NUM_BINS, 0);
    
    std::cout << "=== RANDOM VERİ İLE TEST ===" << std::endl;
    
    // CPU Single Thread
    std::cout << "CPU (Single Thread) hesaplanıyor..." << std::endl;
    double time_cpu_single = HistogramCPU::timeHistogramComputation(
        HistogramCPU::computeHistogramSingle,
        random_data, hist_cpu_single, NUM_BINS, MIN_VALUE, MAX_VALUE
    );
    
    // CPU Parallel
    std::cout << "CPU (Parallel) hesaplanıyor..." << std::endl;
    double time_cpu_parallel = HistogramCPU::timeHistogramComputationParallel(
        random_data, hist_cpu_parallel, NUM_BINS, MIN_VALUE, MAX_VALUE, 8
    );
    
    // CUDA Basic
    std::cout << "CUDA (Basic) hesaplanıyor..." << std::endl;
    double time_cuda_basic = HistogramCUDA::timeHistogramComputation(
        HistogramCUDA::computeHistogramBasic,
        random_data, hist_cuda_basic, NUM_BINS, MIN_VALUE, MAX_VALUE
    );
    
    // CUDA Optimized
    std::cout << "CUDA (Optimized) hesaplanıyor..." << std::endl;
    double time_cuda_optimized = HistogramCUDA::timeHistogramComputation(
        HistogramCUDA::computeHistogramOptimized,
        random_data, hist_cuda_optimized, NUM_BINS, MIN_VALUE, MAX_VALUE
    );
    
    std::cout << std::endl;
    std::cout << "=== PERFORMANS SONUÇLARI ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU (Single Thread):  " << time_cpu_single << " ms" << std::endl;
    std::cout << "CPU (Parallel):       " << time_cpu_parallel << " ms" << std::endl;
    std::cout << "CUDA (Basic):         " << time_cuda_basic << " ms" << std::endl;
    std::cout << "CUDA (Optimized):     " << time_cuda_optimized << " ms" << std::endl;
    std::cout << std::endl;
    
    // Hızlanma oranları
    std::cout << "=== HIZLANMA ORANLARI ===" << std::endl;
    Utils::printPerformanceComparison("CPU Single", time_cpu_single, "CPU Parallel", time_cpu_parallel);
    Utils::printPerformanceComparison("CPU Single", time_cpu_single, "CUDA Basic", time_cuda_basic);
    Utils::printPerformanceComparison("CPU Single", time_cpu_single, "CUDA Optimized", time_cuda_optimized);
    Utils::printPerformanceComparison("CUDA Basic", time_cuda_basic, "CUDA Optimized", time_cuda_optimized);
    std::cout << std::endl;
    
    // Sonuçları doğrula
    std::cout << "=== SONUÇ DOĞRULAMA ===" << std::endl;
    bool cpu_match = Utils::compareHistograms(hist_cpu_single, hist_cpu_parallel);
    bool cuda_basic_match = Utils::compareHistograms(hist_cpu_single, hist_cuda_basic);
    bool cuda_opt_match = Utils::compareHistograms(hist_cpu_single, hist_cuda_optimized);
    
    std::cout << "CPU Single vs CPU Parallel:    " << (cpu_match ? "✓ EŞLEŞIYOR" : "✗ EŞLEŞMIYOR") << std::endl;
    std::cout << "CPU Single vs CUDA Basic:      " << (cuda_basic_match ? "✓ EŞLEŞIYOR" : "✗ EŞLEŞMIYOR") << std::endl;
    std::cout << "CPU Single vs CUDA Optimized:  " << (cuda_opt_match ? "✓ EŞLEŞIYOR" : "✗ EŞLEŞMIYOR") << std::endl;
    std::cout << std::endl;
    
    // Histogram istatistiklerini yazdır
    std::cout << "=== HISTOGRAM İSTATİSTİKLERİ ===" << std::endl;
    Utils::printHistogramStats(hist_cpu_single);
    std::cout << std::endl;
    
    // Histogram görselleştirmesi (sadece ilk 20 bin)
    std::cout << "=== HISTOGRAM ÖRNEĞİ (İlk 20 Bin) ===" << std::endl;
    for (int i = 0; i < 20 && i < NUM_BINS; ++i) {
        std::cout << "Bin " << std::setw(3) << i << ": " << hist_cpu_single[i] << std::endl;
    }
    
    // Gaussian veri ile kısa test
    std::cout << std::endl << "=== GAUSSIAN VERİ İLE KISA TEST ===" << std::endl;
    std::vector<int> hist_gaussian(NUM_BINS, 0);
    double time_gaussian = HistogramCUDA::timeHistogramComputation(
        HistogramCUDA::computeHistogramOptimized,
        gaussian_data, hist_gaussian, NUM_BINS, MIN_VALUE, MAX_VALUE
    );
    std::cout << "Gaussian veri CUDA hesaplama süresi: " << time_gaussian << " ms" << std::endl;
    Utils::printHistogramStats(hist_gaussian);
    
    std::cout << std::endl << "Program başarıyla tamamlandı!" << std::endl;
    
    return 0;
}