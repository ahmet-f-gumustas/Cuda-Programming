#include "../include/image_processor.h"
#include <iostream>
#include <string>
#include <chrono>

void printDeviceInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "CUDA desteği olan cihaz bulunamadı." << std::endl;
        return;
    }
    
    std::cout << "Bulunan CUDA cihaz sayısı: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "\nCihaz " << i << ": " << deviceProp.name << std::endl;
        std::cout << "CUDA Yeteneği: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Toplam Global Bellek: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "Multiprocessor Sayısı: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Maksimum Thread Sayısı: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Maksimum Grid Boyutu: (" 
                 << deviceProp.maxGridSize[0] << ", " 
                 << deviceProp.maxGridSize[1] << ", " 
                 << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "Warp Boyutu: " << deviceProp.warpSize << std::endl;
    }
}

void showMenu() {
    std::cout << "\n===== ParallelVision - CUDA Görüntü İşleme =====" << std::endl;
    std::cout << "1. Görüntü Yükle" << std::endl;
    std::cout << "2. Gri Tonlama (CPU vs GPU)" << std::endl;
    std::cout << "3. Gauss Bulanıklaştırma (CPU vs GPU)" << std::endl;
    std::cout << "4. Sobel Kenar Algılama (CPU vs GPU)" << std::endl;
    std::cout << "5. Histogram Eşitleme (CPU vs GPU)" << std::endl;
    std::cout << "6. Keskinleştirme (CPU vs GPU)" << std::endl;
    std::cout << "7. Tüm İşlemleri Çalıştır ve Performans Karşılaştır" << std::endl;
    std::cout << "0. Çıkış" << std::endl;
    std::cout << "Seçiminiz: ";
}

int main(int argc, char** argv) {
    // GPU bilgilerini göster
    printDeviceInfo();
    
    pv::ImageProcessor processor;
    bool imageLoaded = false;
    std::string imagePath;
    int choice;
    
    do {
        showMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 1: {
                std::cout << "Görüntü yolunu girin: ";
                std::cin >> imagePath;
                imageLoaded = processor.loadImage(imagePath);
                if (imageLoaded) {
                    std::cout << "Görüntü başarıyla yüklendi." << std::endl;
                } else {
                    std::cout << "Görüntü yüklenemedi!" << std::endl;
                }
                break;
            }
            case 2: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat grayCPU = processor.grayscaleCPU(cv::Mat());
                auto end = std::chrono::high_resolution_clock::now();
                auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                cv::Mat grayGPU = processor.grayscaleGPU(cv::Mat());
                end = std::chrono::high_resolution_clock::now();
                auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
                std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
                std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
                
                cv::imshow("Gri Tonlama - CPU", grayCPU);
                cv::imshow("Gri Tonlama - GPU", grayGPU);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            }
            case 3: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                int kernelSize;
                float sigma;
                std::cout << "Kernel boyutu (tek sayı giriniz): ";
                std::cin >> kernelSize;
                std::cout << "Sigma değeri: ";
                std::cin >> sigma;
                
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat blurCPU = processor.gaussianBlurCPU(cv::Mat(), kernelSize, sigma);
                auto end = std::chrono::high_resolution_clock::now();
                auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                cv::Mat blurGPU = processor.gaussianBlurGPU(cv::Mat(), kernelSize, sigma);
                end = std::chrono::high_resolution_clock::now();
                auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
                std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
                std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
                
                cv::imshow("Gauss Bulanıklaştırma - CPU", blurCPU);
                cv::imshow("Gauss Bulanıklaştırma - GPU", blurGPU);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            }
            case 4: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat edgesCPU = processor.sobelEdgeDetectionCPU(cv::Mat());
                auto end = std::chrono::high_resolution_clock::now();
                auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                cv::Mat edgesGPU = processor.sobelEdgeDetectionGPU(cv::Mat());
                end = std::chrono::high_resolution_clock::now();
                auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
                std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
                std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
                
                cv::imshow("Sobel Kenar Algılama - CPU", edgesCPU);
                cv::imshow("Sobel Kenar Algılama - GPU", edgesGPU);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            }
            case 5: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat histEqCPU = processor.histogramEqualizationCPU(cv::Mat());
                auto end = std::chrono::high_resolution_clock::now();
                auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                cv::Mat histEqGPU = processor.histogramEqualizationGPU(cv::Mat());
                end = std::chrono::high_resolution_clock::now();
                auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
                std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
                std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
                
                cv::imshow("Histogram Eşitleme - CPU", histEqCPU);
                cv::imshow("Histogram Eşitleme - GPU", histEqGPU);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            }
            case 6: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                auto start = std::chrono::high_resolution_clock::now();
                cv::Mat sharpenCPU = processor.sharpenCPU(cv::Mat());
                auto end = std::chrono::high_resolution_clock::now();
                auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                start = std::chrono::high_resolution_clock::now();
                cv::Mat sharpenGPU = processor.sharpenGPU(cv::Mat());
                end = std::chrono::high_resolution_clock::now();
                auto gpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                
                std::cout << "CPU Süre: " << cpuDuration << " ms" << std::endl;
                std::cout << "GPU Süre: " << gpuDuration << " ms" << std::endl;
                std::cout << "Hızlanma: " << (float)cpuDuration / gpuDuration << "x" << std::endl;
                
                cv::imshow("Keskinleştirme - CPU", sharpenCPU);
                cv::imshow("Keskinleştirme - GPU", sharpenGPU);
                cv::waitKey(0);
                cv::destroyAllWindows();
                break;
            }
            case 7: {
                if (!imageLoaded) {
                    std::cout << "Önce bir görüntü yükleyin!" << std::endl;
                    break;
                }
                
                processor.runPerformanceTest();
                break;
            }
            case 0:
                std::cout << "Programdan çıkılıyor..." << std::endl;
                break;
            default:
                std::cout << "Geçersiz seçenek!" << std::endl;
        }
        
    } while (choice != 0);
    
    return 0;
}
