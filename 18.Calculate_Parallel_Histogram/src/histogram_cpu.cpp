#include "histogram_cpu.h"
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>

void HistogramCPU::computeHistogramSingle(
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value
) {
    // Histogram'ı sıfırla
    std::fill(histogram.begin(), histogram.end(), 0);
    
    // Her veri noktası için bin hesapla
    int range = max_value - min_value + 1;
    double bin_width = static_cast<double>(range) / num_bins;
    
    for (const int& value : data) {
        // Değer aralık dışındaysa atla
        if (value < min_value || value > max_value) {
            continue;
        }
        
        // Hangi bin'e ait olduğunu hesapla
        int bin_index = static_cast<int>((value - min_value) / bin_width);
        
        // Son bin için sınır kontrolü
        if (bin_index >= num_bins) {
            bin_index = num_bins - 1;
        }
        
        histogram[bin_index]++;
    }
}

void HistogramCPU::computeHistogramParallel(
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value,
    int num_threads
) {
    // Histogram'ı sıfırla
    std::fill(histogram.begin(), histogram.end(), 0);
    
    // Her thread için yerel histogram
    std::vector<std::vector<int>> local_histograms(num_threads, std::vector<int>(num_bins, 0));
    
    // Thread'ler
    std::vector<std::thread> threads;
    
    int range = max_value - min_value + 1;
    double bin_width = static_cast<double>(range) / num_bins;
    
    // Her thread için iş bölümü
    int chunk_size = data.size() / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        int start_idx = t * chunk_size;
        int end_idx = (t == num_threads - 1) ? data.size() : (t + 1) * chunk_size;
        
        threads.emplace_back([&, t, start_idx, end_idx]() {
            for (int i = start_idx; i < end_idx; ++i) {
                int value = data[i];
                
                // Değer aralık dışındaysa atla
                if (value < min_value || value > max_value) {
                    continue;
                }
                
                // Hangi bin'e ait olduğunu hesapla
                int bin_index = static_cast<int>((value - min_value) / bin_width);
                
                // Son bin için sınır kontrolü
                if (bin_index >= num_bins) {
                    bin_index = num_bins - 1;
                }
                
                local_histograms[t][bin_index]++;
            }
        });
    }
    
    // Thread'lerin bitmesini bekle
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Yerel histogram'ları birleştir
    for (int t = 0; t < num_threads; ++t) {
        for (int bin = 0; bin < num_bins; ++bin) {
            histogram[bin] += local_histograms[t][bin];
        }
    }
}

double HistogramCPU::timeHistogramComputation(
    void (*compute_func)(const std::vector<int>&, std::vector<int>&, int, int, int),
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    compute_func(data, histogram, num_bins, min_value, max_value);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0; // millisecond cinsinden döndür
}

double HistogramCPU::timeHistogramComputationParallel(
    const std::vector<int>& data, 
    std::vector<int>& histogram, 
    int num_bins,
    int min_value, 
    int max_value,
    int num_threads
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    computeHistogramParallel(data, histogram, num_bins, min_value, max_value, num_threads);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0; // millisecond cinsinden döndür
}