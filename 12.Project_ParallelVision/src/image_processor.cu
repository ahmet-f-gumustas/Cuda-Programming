// Bu dosya `.cu` uzantılı olmalı
#include "../include/image_processor.h"
#include "../include/cuda_kernels.cuh"
#include <iostream>

namespace pv {

ImageProcessor::ImageProcessor() 
    : d_input(nullptr), d_output(nullptr), 
      imageWidth(0), imageHeight(0), imageChannels(0), imageSize(0) {
}

ImageProcessor::~ImageProcessor() {
    freeMemory();
}

bool ImageProcessor::loadImage(const std::string& filename) {
    originalImage = cv::imread(filename, cv::IMREAD_COLOR);
    if (originalImage.empty()) {
        std::cerr << "Görüntü yüklenemedi: " << filename << std::endl;
        return false;
    }
    
    imageWidth = originalImage.cols;
    imageHeight = originalImage.rows;
    imageChannels = originalImage.channels();
    imageSize = imageWidth * imageHeight * imageChannels;
    
    // GPU belleği ayrılıyor
    allocateMemory(imageWidth, imageHeight);
    
    return true;
}

bool ImageProcessor::saveImage(const std::string& filename, const cv::Mat& image) {
    return cv::imwrite(filename, image);
}

void ImageProcessor::allocateMemory(int width, int height) {
    // Önceki belleği temizle
    freeMemory();
    
    // Yeni boyutlar ayarlanıyor
    imageWidth = width;
    imageHeight = height;
    imageChannels = originalImage.channels();
    imageSize = width * height * imageChannels;
    
    // CUDA belleği ayrılıyor
    CUDA_CHECK_ERROR(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK_ERROR(cudaMalloc(&d_output, imageSize));
}

void ImageProcessor::freeMemory() {
    if (d_input) {
        cudaFree(d_input);
        d_input = nullptr;
    }
    
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
}

void ImageProcessor::runPerformanceTest() {
    std::cout << "\n===== Performans Karşılaştırması =====" << std::endl;
    
    // Gri tonlama testi
    std::cout << "\n1. Gri Tonlama:" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat grayCPU = grayscaleCPU(originalImage);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat grayGPU = grayscaleGPU(originalImage);
    end = std::chrono::high_resolution_clock::now();
    auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
    std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
    
    // Gauss bulanıklaştırma testi
    std::cout << "\n2. Gauss Bulanıklaştırma:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat blurCPU = gaussianBlurCPU(originalImage, 5, 1.0f);
    end = std::chrono::high_resolution_clock::now();
    cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat blurGPU = gaussianBlurGPU(originalImage, 5, 1.0f);
    end = std::chrono::high_resolution_clock::now();
    gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
    std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
    
    // Sobel kenar algılama testi
    std::cout << "\n3. Sobel Kenar Algılama:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat edgesCPU = sobelEdgeDetectionCPU(grayCPU);
    end = std::chrono::high_resolution_clock::now();
    cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat edgesGPU = sobelEdgeDetectionGPU(grayGPU);
    end = std::chrono::high_resolution_clock::now();
    gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
    std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
    
    // Histogram eşitleme testi
    std::cout << "\n4. Histogram Eşitleme:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat histEqCPU = histogramEqualizationCPU(grayCPU);
    end = std::chrono::high_resolution_clock::now();
    cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat histEqGPU = histogramEqualizationGPU(grayGPU);
    end = std::chrono::high_resolution_clock::now();
    gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
    std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
    
    // Keskinleştirme testi
    std::cout << "\n5. Keskinleştirme:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cv::Mat sharpenCPUResult = sharpenCPU(originalImage);  // sharpenCPU() yerine farklı bir isim kullanıyorum
    end = std::chrono::high_resolution_clock::now();
    cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    cv::Mat sharpenGPUResult = sharpenGPU(originalImage);  // sharpenGPU() yerine farklı bir isim kullanıyorum
    end = std::chrono::high_resolution_clock::now();
    gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
    std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
    std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
    
    // Sonuçları göster
    cv::imshow("Orijinal Görüntü", originalImage);
    cv::imshow("Gri Tonlama - CPU", grayCPU);
    cv::imshow("Gri Tonlama - GPU", grayGPU);
    cv::imshow("Gauss Bulanıklaştırma - CPU", blurCPU);
    cv::imshow("Gauss Bulanıklaştırma - GPU", blurGPU);
    cv::imshow("Sobel Kenar Algılama - CPU", edgesCPU);
    cv::imshow("Sobel Kenar Algılama - GPU", edgesGPU);
    cv::imshow("Histogram Eşitleme - CPU", histEqCPU);
    cv::imshow("Histogram Eşitleme - GPU", histEqGPU);
    cv::imshow("Keskinleştirme - CPU", sharpenCPUResult);  // değişken ismi değişti
    cv::imshow("Keskinleştirme - GPU", sharpenGPUResult);  // değişken ismi değişti
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// CPU implementasyonları
cv::Mat ImageProcessor::grayscaleCPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat grayImage;
    
    cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    return grayImage;
}

cv::Mat ImageProcessor::gaussianBlurCPU(const cv::Mat& input, int kernelSize, float sigma) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat blurredImage;
    
    cv::GaussianBlur(img, blurredImage, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return blurredImage;
}

cv::Mat ImageProcessor::sobelEdgeDetectionCPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? grayscaleCPU(originalImage) : input;
    cv::Mat grayImage;
    
    // Görüntü gri tonlama değilse, önce gri tonlamaya dönüştür
    if (img.channels() > 1) {
        cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = img.clone();
    }
    
    cv::Mat gradX, gradY, absGradX, absGradY, result;
    
    // Sobel operatörü uygula
    cv::Sobel(grayImage, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(grayImage, gradY, CV_16S, 0, 1, 3);
    
    // Mutlak değer
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    
    // Gradyanları birleştir
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, result);
    
    return result;
}

cv::Mat ImageProcessor::histogramEqualizationCPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? grayscaleCPU(originalImage) : input;
    cv::Mat grayImage, result;
    
    // Görüntü gri tonlama değilse, önce gri tonlamaya dönüştür
    if (img.channels() > 1) {
        cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = img.clone();
    }
    
    // Histogram eşitleme uygula
    cv::equalizeHist(grayImage, result);
    
    return result;
}

cv::Mat ImageProcessor::sharpenCPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat result;
    
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1);
    
    cv::filter2D(img, result, -1, kernel);
    
    return result;
}

// GPU implementasyonları
cv::Mat ImageProcessor::grayscaleGPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat result(img.rows, img.cols, CV_8UC1);
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, img.data, imageSize, cudaMemcpyHostToDevice));
    
    // Thread ve block boyutlarını ayarla
    dim3 blockSize(16, 16);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, 
                  (img.rows + blockSize.y - 1) / blockSize.y);
    
    // CUDA kernel'i çalıştır
    grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, img.cols, img.rows, img.channels());
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Sonucu geri al
    CUDA_CHECK_ERROR(cudaMemcpy(result.data, d_output, img.cols * img.rows, cudaMemcpyDeviceToHost));
    
    return result;
}

cv::Mat ImageProcessor::gaussianBlurGPU(const cv::Mat& input, int kernelSize, float sigma) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat result;
    
    if (img.channels() == 1) {
        result = cv::Mat(img.rows, img.cols, CV_8UC1);
    } else {
        result = cv::Mat(img.rows, img.cols, CV_8UC3);
    }
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, img.data, imageSize, cudaMemcpyHostToDevice));
    
    // Gaussian kernel oluştur
    float* d_kernel;
    CUDA_CHECK_ERROR(cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float)));
    
    dim3 blockSizeKernel(kernelSize, kernelSize);
    generateGaussianKernel<<<1, blockSizeKernel>>>(d_kernel, kernelSize, sigma);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Thread ve block boyutlarını ayarla
    dim3 blockSize(16, 16);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, 
                  (img.rows + blockSize.y - 1) / blockSize.y);
    
    // CUDA kernel'i çalıştır
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, 
                                               img.cols, img.rows, kernelSize);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Sonucu geri al
    CUDA_CHECK_ERROR(cudaMemcpy(result.data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Temporary kernel belleğini temizle
    cudaFree(d_kernel);
    
    return result;
}

cv::Mat ImageProcessor::sobelEdgeDetectionGPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? grayscaleCPU(originalImage) : input;
    cv::Mat grayImage;
    
    // Görüntü gri tonlama değilse, önce gri tonlamaya dönüştür
    if (img.channels() > 1) {
        cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = img.clone();
    }
    
    cv::Mat result(grayImage.rows, grayImage.cols, CV_8UC1);
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, grayImage.data, grayImage.cols * grayImage.rows, cudaMemcpyHostToDevice));
    
    // Thread ve block boyutlarını ayarla
    dim3 blockSize(16, 16);
    dim3 gridSize((grayImage.cols + blockSize.x - 1) / blockSize.x, 
                  (grayImage.rows + blockSize.y - 1) / blockSize.y);
    
    // CUDA kernel'i çalıştır
    sobelEdgeDetectionKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                     grayImage.cols, grayImage.rows);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Sonucu geri al
    CUDA_CHECK_ERROR(cudaMemcpy(result.data, d_output, grayImage.cols * grayImage.rows, cudaMemcpyDeviceToHost));
    
    return result;
}

cv::Mat ImageProcessor::histogramEqualizationGPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? grayscaleCPU(originalImage) : input;
    cv::Mat grayImage;
    
    // Görüntü gri tonlama değilse, önce gri tonlamaya dönüştür
    if (img.channels() > 1) {
        cv::cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = img.clone();
    }
    
    cv::Mat result(grayImage.rows, grayImage.cols, CV_8UC1);
    
    // Histogram ve CDF için bellek ayrılıyor
    unsigned int* d_histogram;
    unsigned int* d_cdf;
    CUDA_CHECK_ERROR(cudaMalloc(&d_histogram, 256 * sizeof(unsigned int)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_cdf, 256 * sizeof(unsigned int)));
    CUDA_CHECK_ERROR(cudaMemset(d_histogram, 0, 256 * sizeof(unsigned int)));
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, grayImage.data, grayImage.cols * grayImage.rows, cudaMemcpyHostToDevice));
    
    // Thread ve block boyutlarını ayarla
    dim3 blockSize(256);
    dim3 gridSize((grayImage.cols * grayImage.rows + blockSize.x - 1) / blockSize.x);
    
    // Histogram hesapla
    calcHistogramKernel<<<gridSize, blockSize>>>(d_input, d_histogram, 
                                                grayImage.cols, grayImage.rows);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // CPU'da CDF hesapla (bu kısım genellikle GPU'da verimsiz)
    unsigned int histogram[256];
    unsigned int cdf[256];
    CUDA_CHECK_ERROR(cudaMemcpy(histogram, d_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
    
    // CDF'yi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_cdf, cdf, 256 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Histogram eşitleme uygula
    float scale = 255.0f / (grayImage.cols * grayImage.rows);
    
    blockSize = dim3(16, 16);
    gridSize = dim3((grayImage.cols + blockSize.x - 1) / blockSize.x, 
                    (grayImage.rows + blockSize.y - 1) / blockSize.y);
    
    histogramEqualizationKernel<<<gridSize, blockSize>>>(d_input, d_output, 
                                                       grayImage.cols, grayImage.rows, 
                                                       d_histogram, d_cdf, scale);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Sonucu geri al
    CUDA_CHECK_ERROR(cudaMemcpy(result.data, d_output, grayImage.cols * grayImage.rows, cudaMemcpyDeviceToHost));
    
    // Temporary belleği temizle
    cudaFree(d_histogram);
    cudaFree(d_cdf);
    
    return result;
}

cv::Mat ImageProcessor::sharpenGPU(const cv::Mat& input) {
    cv::Mat img = input.empty() ? originalImage : input;
    cv::Mat result;
    
    if (img.channels() == 1) {
        result = cv::Mat(img.rows, img.cols, CV_8UC1);
    } else {
        result = cv::Mat(img.rows, img.cols, CV_8UC3);
    }
    
    // Veriyi GPU'ya kopyala
    CUDA_CHECK_ERROR(cudaMemcpy(d_input, img.data, imageSize, cudaMemcpyHostToDevice));
    
    // Thread ve block boyutlarını ayarla
    dim3 blockSize(16, 16);
    dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x, 
                  (img.rows + blockSize.y - 1) / blockSize.y);
    
    // CUDA kernel'i çalıştır
    sharpenKernel<<<gridSize, blockSize>>>(d_input, d_output, img.cols, img.rows);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // Sonucu geri al
    CUDA_CHECK_ERROR(cudaMemcpy(result.data, d_output, imageSize, cudaMemcpyDeviceToHost));
    
    return result;
}

} // namespace pv