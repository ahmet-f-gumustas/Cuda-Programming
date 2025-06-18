#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

class Utils {
public:
    // Test verisi oluşturma
    static std::vector<int> generateRandomData(int size, int min_val = 0, int max_val = 255);
    static std::vector<int> generateGaussianData(int size, double mean = 128.0, double std_dev = 30.0);
    static std::vector<int> generateUniformData(int size, int min_val = 0, int max_val = 255);
    
    // Histogram sonuçlarını karşılaştırma
    static bool compareHistograms(const std::vector<int>& hist1, const std::vector<int>& hist2, double tolerance = 0.01);
    
    // Histogram yazdırma ve kaydetme
    static void printHistogram(const std::vector<int>& histogram, const std::string& title = "Histogram");
    static void saveHistogramToFile(const std::vector<int>& histogram, const std::string& filename);
    static void printHistogramStats(const std::vector<int>& histogram);
    
    // Performans karşılaştırması
    static void printPerformanceComparison(
        const std::string& method1_name, double time1,
        const std::string& method2_name, double time2
    );
    
    // GPU bilgilerini yazdırma
    static void printGPUInfo();
    
    // Veri istatistikleri
    static void printDataStats(const std::vector<int>& data);
};

#endif // UTILS_H