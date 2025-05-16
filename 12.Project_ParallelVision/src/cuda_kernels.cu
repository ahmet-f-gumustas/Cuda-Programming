#include "cuda_kernels.cuh"
#include <cmath>

namespace pv {

// Gri tonlama kernel fonksiyonu
__global__ void grayscaleKernel(const unsigned char* input, unsigned char* output, 
                               int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        int outputIdx = y * width + x;
        
        // BGR değerleri
        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];
        
        // Gri tonlama formülü (0.299 * R + 0.587 * G + 0.114 * B)
        output[outputIdx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Gauss bulanıklaştırma kernel fonksiyonları
__global__ void generateGaussianKernel(float* kernel, int kernelSize, float sigma) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    
    if (x < kernelSize && y < kernelSize) {
        int center = kernelSize / 2;
        float dx = x - center;
        float dy = y - center;
        float exponent = -(dx * dx + dy * dy) / (2 * sigma * sigma);
        kernel[y * kernelSize + x] = exp(exponent) / (2 * M_PI * sigma * sigma);
    }
}

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, 
                                 float* kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        float kernelSum = 0.0f;
        int halfKernelSize = kernelSize / 2;
        
        for (int ky = -halfKernelSize; ky <= halfKernelSize; ++ky) {
            for (int kx = -halfKernelSize; kx <= halfKernelSize; ++kx) {
                int i = y + ky;
                int j = x + kx;
                
                // Sınırları kontrol et (mirror padding)
                if (i < 0) i = -i;
                if (i >= height) i = 2 * height - i - 2;
                if (j < 0) j = -j;
                if (j >= width) j = 2 * width - j - 2;
                
                int kernelIdx = (ky + halfKernelSize) * kernelSize + (kx + halfKernelSize);
                float kernelValue = kernel[kernelIdx];
                
                // 3 kanallı BGR formatı
                int idx = (i * width + j) * 3;
                sum.x += input[idx] * kernelValue;        // B
                sum.y += input[idx + 1] * kernelValue;    // G
                sum.z += input[idx + 2] * kernelValue;    // R
                
                kernelSum += kernelValue;
            }
        }
        
        // Normalizasyon
        int outputIdx = (y * width + x) * 3;
        output[outputIdx] = static_cast<unsigned char>(sum.x / kernelSum);        // B
        output[outputIdx + 1] = static_cast<unsigned char>(sum.y / kernelSum);    // G
        output[outputIdx + 2] = static_cast<unsigned char>(sum.z / kernelSum);    // R
    }
}

// Sobel kenar algılama kernel fonksiyonu
__global__ void sobelEdgeDetectionKernel(const unsigned char* input, unsigned char* output, 
                                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel operatörleri
        int gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        
        int gy[3][3] = {
            {-1, -2, -1},
            {0,  0,  0},
            {1,  2,  1}
        };
        
        int sumX = 0;
        int sumY = 0;
        
        // 3x3 penceresi üzerinde konvolüsyon
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int idx = (y + i) * width + (x + j);
                int pixel = input[idx];
                
                sumX += pixel * gx[i+1][j+1];
                sumY += pixel * gy[i+1][j+1];
            }
        }
        
        // Gradyanın büyüklüğünü hesapla
        float magnitude = sqrtf(sumX * sumX + sumY * sumY);
        
        // Sonucu 0-255 aralığına sınırlandır
        output[y * width + x] = min(255, max(0, static_cast<int>(magnitude)));
    }
    else if (x < width && y < height) {
        // Kenarlara sıfır değeri ata
        output[y * width + x] = 0;
    }
}

// Histogram hesaplama kernel fonksiyonları
__global__ void calcHistogramKernel(const unsigned char* input, unsigned int* histogram, 
                                  int width, int height) {
    __shared__ unsigned int sharedHist[256];
    
    int idx = threadIdx.x;
    if (idx < 256) {
        sharedHist[idx] = 0;
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < width * height) {
        unsigned char pixel = input[i];
        atomicAdd(&sharedHist[pixel], 1);
    }
    __syncthreads();
    
    if (idx < 256) {
        atomicAdd(&histogram[idx], sharedHist[idx]);
    }
}

__global__ void histogramEqualizationKernel(const unsigned char* input, unsigned char* output, 
                                          int width, int height, unsigned int* histogram, 
                                          unsigned int* cdf, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char pixel = input[idx];
        
        // CDF değerini kullanarak yeni piksel değerini hesapla
        unsigned int cdfValue = cdf[pixel];
        unsigned char newPixel = static_cast<unsigned char>(cdfValue * scale);
        
        output[idx] = newPixel;
    }
}

// Keskinleştirme kernel fonksiyonu
__global__ void sharpenKernel(const unsigned char* input, unsigned char* output, 
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Keskinleştirme filtresi
        int kernel[3][3] = {
            {-1, -1, -1},
            {-1,  9, -1},
            {-1, -1, -1}
        };
        
        for (int c = 0; c < 3; ++c) {
            int sum = 0;
            
            // 3x3 penceresi üzerinde konvolüsyon
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int idx = ((y + i) * width + (x + j)) * 3 + c;
                    int pixel = input[idx];
                    
                    sum += pixel * kernel[i+1][j+1];
                }
            }
            
            // Sonucu 0-255 aralığına sınırlandır
            int outputIdx = (y * width + x) * 3 + c;
            output[outputIdx] = min(255, max(0, sum));
        }
    }
    else if (x < width && y < height) {
        // Kenar piksellerini kopyala
        int idx = (y * width + x) * 3;
        int outputIdx = idx;
        
        output[outputIdx] = input[idx];
        output[outputIdx + 1] = input[idx + 1];
        output[outputIdx + 2] = input[idx + 2];
    }
}

} // namespace pv
