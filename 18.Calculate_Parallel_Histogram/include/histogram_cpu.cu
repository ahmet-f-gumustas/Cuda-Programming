#ifndef HISTOGRAM_CPU_H
#define HISTOGRAM_CPU_H

#include <vector>
#include <chrono>

class HistogramCPU {
public:
    // CPU üzerinde histogram hesaplama (tek thread)
    static void computeHistogramSingle(
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255
    );
    
    // CPU üzerinde histogram hesaplama (çoklu thread)
    static void computeHistogramParallel(
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255,
        int num_threads = 4
    );
    
    // Performans ölçümü için wrapper fonksiyon
    static double timeHistogramComputation(
        void (*compute_func)(const std::vector<int>&, std::vector<int>&, int, int, int),
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255
    );
    
    // Çoklu thread versiyonu için özel wrapper
    static double timeHistogramComputationParallel(
        const std::vector<int>& data, 
        std::vector<int>& histogram, 
        int num_bins,
        int min_value = 0, 
        int max_value = 255,
        int num_threads = 4
    );
};

#endif // HISTOGRAM_CPU_H