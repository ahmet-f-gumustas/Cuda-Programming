#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

int main(int argc, char** argv) {
    // CUDA kullanılabilirliğini kontrol et
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices == 0) {
        std::cerr << "CUDA destekleyen GPU bulunamadı!" << std::endl;
        return -1;
    }
    
    std::cout << "Kullanılabilir CUDA cihazı sayısı: " << cuda_devices << std::endl;
    
    // CUDA cihaz bilgilerini yazdır
    for (int dev = 0; dev < cuda_devices; ++dev) {
        cv::cuda::printCudaDeviceInfo(dev);
    }
    
    // Örnek bir görüntü yükle
    std::string imagePath = "input.jpg";
    if (argc > 1) {
        imagePath = argv[1];
    }
    
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Görüntü yüklenemedi: " << imagePath << std::endl;
        return -1;
    }
    
    // Görüntü boyutlarını yazdır
    std::cout << "Görüntü boyutları: " << img.cols << "x" << img.rows << std::endl;
    
    // CPU kullanarak gri tonlamaya dönüştür ve süreyi ölç
    cv::Mat cpu_gray;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(img, cpu_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(cpu_gray, cpu_gray, cv::Size(5, 5), 0);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    // GPU'ya görüntüyü yükle
    cv::cuda::GpuMat gpu_img;
    gpu_img.upload(img);
    
    // CUDA kullanarak gri tonlamaya dönüştür ve süreyi ölç
    cv::cuda::GpuMat gpu_gray;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY);
    
    // CUDA ile Gaussian bulanıklaştırma filtresi oluştur ve uygula
    cv::Ptr<cv::cuda::Filter> gaussian_filter = cv::cuda::createGaussianFilter(
        gpu_gray.type(), gpu_gray.type(), cv::Size(5, 5), 0);
    gaussian_filter->apply(gpu_gray, gpu_gray);
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
    
    // GPU'dan sonucu al
    cv::Mat result;
    gpu_gray.download(result);
    
    // Sonuçları yazdır
    std::cout << "CPU işlem süresi: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "GPU işlem süresi: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "Hızlanma oranı: " << static_cast<float>(cpu_duration.count()) / gpu_duration.count() << "x" << std::endl;
    
    // Sonuçları görüntüle ve kaydet
    cv::imwrite("cpu_result.jpg", cpu_gray);
    cv::imwrite("gpu_result.jpg", result);
    
    // Görüntüleri göster
    cv::imshow("Orijinal Görüntü", img);
    cv::imshow("CPU Gri Tonlama + Gaussian", cpu_gray);
    cv::imshow("GPU Gri Tonlama + Gaussian", result);
    cv::waitKey(0);
    
    return 0;
}