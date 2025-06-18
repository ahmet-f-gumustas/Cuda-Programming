#include "utils.h"
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

std::vector<int> Utils::generateRandomData(int size, int min_val, int max_val) {
    std::vector<int> data;
    data.reserve(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data.push_back(dis(gen));
    }
    
    return data;
}

std::vector<int> Utils::generateGaussianData(int size, double mean, double std_dev) {
    std::vector<int> data;
    data.reserve(size);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, std_dev);
    
    for (int i = 0; i < size; ++i) {
        double value = dis(gen);
        // 0-255 aralığına sınırla
        int int_value = static_cast<int>(std::round(value));
        int_value = std::max(0, std::min(255, int_value));
        data.push_back(int_value);
    }
    
    return data;
}

std::vector<int> Utils::generateUniformData(int size, int min_val, int max_val) {
    std::vector<int> data;
    data.reserve(size);
    
    for (int i = 0; i < size; ++i) {
        data.push_back(min_val + (i % (max_val - min_val + 1)));
    }
    
    // Karıştır
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(data.begin(), data.end(), gen);
    
    return data;
}

bool Utils::compareHistograms(const std::vector<int>& hist1, const std::vector<int>& hist2, double tolerance) {
    if (hist1.size() != hist2.size()) {
        return false;
    }
    
    // Toplam değerleri hesapla
    long long sum1 = std::accumulate(hist1.begin(), hist1.end(), 0LL);
    long long sum2 = std::accumulate(hist2.begin(), hist2.end(), 0LL);
    
    if (sum1 != sum2) {
        std::cout << "Toplam değerler farklı: " << sum1 << " vs " << sum2 << std::endl;
        return false;
    }
    
    // Bin-by-bin karşılaştır
    for (size_t i = 0; i < hist1.size(); ++i) {
        if (hist1[i] != hist2[i]) {
            double diff = std::abs(hist1[i] - hist2[i]);
            double relative_diff = (sum1 > 0) ? diff / static_cast<double>(sum1) : 0.0;
            
            if (relative_diff > tolerance) {
                std::cout << "Bin " << i << " farklı: " << hist1[i] << " vs " << hist2[i] 
                         << " (fark: " << diff << ", relatif fark: " << relative_diff << ")" << std::endl;
                return false;
            }
        }
    }
    
    return true;
}

void Utils::printHistogram(const std::vector<int>& histogram, const std::string& title) {
    std::cout << title << ":" << std::endl;
    for (size_t i = 0; i < histogram.size(); ++i) {
        std::cout << "Bin " << std::setw(3) << i << ": " << histogram[i] << std::endl;
    }
    std::cout << std::endl;
}

void Utils::saveHistogramToFile(const std::vector<int>& histogram, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Bin,Count\n";
        for (size_t i = 0; i < histogram.size(); ++i) {
            file << i << "," << histogram[i] << "\n";
        }
        file.close();
        std::cout << "Histogram " << filename << " dosyasına kaydedildi." << std::endl;
    } else {
        std::cerr << "Dosya açılamadı: " << filename << std::endl;
    }
}

void Utils::printHistogramStats(const std::vector<int>& histogram) {
    // Temel istatistikler
    long long total_count = std::accumulate(histogram.begin(), histogram.end(), 0LL);
    int max_count = *std::max_element(histogram.begin(), histogram.end());
    int min_count = *std::min_element(histogram.begin(), histogram.end());
    
    // En popüler bin
    auto max_it = std::max_element(histogram.begin(), histogram.end());
    int max_bin = std::distance(histogram.begin(), max_it);
    
    // Ortalama ve standart sapma
    double mean = static_cast<double>(total_count) / histogram.size();
    double variance = 0.0;
    for (int count : histogram) {
        variance += std::pow(count - mean, 2);
    }
    variance /= histogram.size();
    double std_dev = std::sqrt(variance);
    
    std::cout << "Histogram İstatistikleri:" << std::endl;
    std::cout << "- Toplam sayım: " << total_count << std::endl;
    std::cout << "- Bin sayısı: " << histogram.size() << std::endl;
    std::cout << "- Maksimum sayım: " << max_count << " (Bin " << max_bin << ")" << std::endl;
    std::cout << "- Minimum sayım: " << min_count << std::endl;
    std::cout << "- Ortalama sayım: " << std::fixed << std::setprecision(2) << mean << std::endl;
    std::cout << "- Standart sapma: " << std::fixed << std::setprecision(2) << std_dev << std::endl;
}

void Utils::printPerformanceComparison(
    const std::string& method1_name, double time1,
    const std::string& method2_name, double time2
) {
    double speedup = time1 / time2;
    std::cout << std::fixed << std::setprecision(2);
    
    if (speedup > 1.0) {
        std::cout << method2_name << ", " << method1_name << "'dan " 
                  << speedup << "x daha hızlı" << std::endl;
    } else {
        std::cout << method1_name << ", " << method2_name << "'dan " 
                  << (1.0/speedup) << "x daha hızlı" << std::endl;
    }
}

void Utils::printGPUInfo() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "=== GPU BİLGİLERİ ===" << std::endl;
    std::cout << "GPU sayısı: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "- Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "- Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB" << std::endl;
        std::cout << "- Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
        std::cout << "- Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "- Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "- Max Grid Size: " << prop.maxGridSize[0] << " x " 
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "- Warp Size: " << prop.warpSize << std::endl;
        std::cout << "- Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
        std::cout << "- Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    }
}

void Utils::printDataStats(const std::vector<int>& data) {
    if (data.empty()) {
        std::cout << "Veri boş!" << std::endl;
        return;
    }
    
    // Temel istatistikler
    auto minmax = std::minmax_element(data.begin(), data.end());
    int min_val = *minmax.first;
    int max_val = *minmax.second;
    
    long long sum = std::accumulate(data.begin(), data.end(), 0LL);
    double mean = static_cast<double>(sum) / data.size();
    
    // Standart sapma
    double variance = 0.0;
    for (int value : data) {
        variance += std::pow(value - mean, 2);
    }
    variance /= data.size();
    double std_dev = std::sqrt(variance);
    
    // Medyan (sorted copy gerekli)
    std::vector<int> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    double median;
    if (sorted_data.size() % 2 == 0) {
        median = (sorted_data[sorted_data.size()/2 - 1] + sorted_data[sorted_data.size()/2]) / 2.0;
    } else {
        median = sorted_data[sorted_data.size()/2];
    }
    
    std::cout << "Veri İstatistikleri:" << std::endl;
    std::cout << "- Eleman sayısı: " << data.size() << std::endl;
    std::cout << "- Minimum: " << min_val << std::endl;
    std::cout << "- Maksimum: " << max_val << std::endl;
    std::cout << "- Ortalama: " << std::fixed << std::setprecision(2) << mean << std::endl;
    std::cout << "- Medyan: " << std::fixed << std::setprecision(2) << median << std::endl;
    std::cout << "- Standart sapma: " << std::fixed << std::setprecision(2) << std_dev << std::endl;
}