#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace pv {

// Gri tonlama kernel fonksiyonu
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, 
                               int width, int height, int channels);

// Gauss bulanıklaştırma kernel fonksiyonları
__global__ void generateGaussianKernel(float* kernel, int kernelSize, float sigma);
__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, 
                                 float* kernel, int width, int height, int kernelSize);

// Sobel kenar algılama kernel fonksiyonu
__global__ void sobelEdgeDetectionKernel(const unsigned char* input, unsigned char* output, 
                                       int width, int height);

// Histogram hesaplama kernel fonksiyonları
__global__ void calcHistogramKernel(const unsigned char* input, unsigned int* histogram, 
                                  int width, int height);
__global__ void histogramEqualizationKernel(const unsigned char* input, unsigned char* output, 
                                          int width, int height, unsigned int* histogram, 
                                          unsigned int* cdf, float scale);

// Keskinleştirme kernel fonksiyonu
__global__ void sharpenKernel(const unsigned char* input, unsigned char* output, 
                             int width, int height);

} // namespace pv
