#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

// Hata kontrolü için yardımcı makro
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

namespace pv {

// Filtrelerin CPU ve GPU implementasyonlarını içeren sınıf
class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Görüntü yükleme ve kaydetme
    bool loadImage(const std::string& filename);
    bool saveImage(const std::string& filename, const cv::Mat& image);

    // Performans ölçüm
    void runPerformanceTest();

    // CPU implementasyonları
    cv::Mat grayscaleCPU(const cv::Mat& input);
    cv::Mat gaussianBlurCPU(const cv::Mat& input, int kernelSize, float sigma);
    cv::Mat sobelEdgeDetectionCPU(const cv::Mat& input);
    cv::Mat histogramEqualizationCPU(const cv::Mat& input);
    cv::Mat sharpenCPU(const cv::Mat& input);

    // GPU implementasyonları
    cv::Mat grayscaleGPU(const cv::Mat& input);
    cv::Mat gaussianBlurGPU(const cv::Mat& input, int kernelSize, float sigma);
    cv::Mat sobelEdgeDetectionGPU(const cv::Mat& input);
    cv::Mat histogramEqualizationGPU(const cv::Mat& input);
    cv::Mat sharpenGPU(const cv::Mat& input);

private:
    cv::Mat originalImage;
    
    // CUDA yardımcı fonksiyonları
    void allocateMemory(int width, int height);
    void freeMemory();

    // CUDA bellek alanları
    unsigned char* d_input;
    unsigned char* d_output;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    size_t imageSize;
};

} // namespace pv
