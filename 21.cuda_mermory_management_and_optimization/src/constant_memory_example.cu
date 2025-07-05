#include "memory_manager.cuh"

// Constant memory - 64KB limit
__constant__ float d_constData[16384]; // 64KB / 4 bytes = 16384 floats

// Constant memory kullanan convolution kernel
__global__ void convolutionConstantMemory(float* output, const float* input, int width, int height, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int halfFilter = filterSize / 2;
        
        for (int fy = -halfFilter; fy <= halfFilter; fy++) {
            for (int fx = -halfFilter; fx <= halfFilter; fx++) {
                int px = x + fx;
                int py = y + fy;
                
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int filterIdx = (fy + halfFilter) * filterSize + (fx + halfFilter);
                    sum += input[py * width + px] * d_constData[filterIdx];
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}

// Global memory kullanan versiyon (karşılaştırma için)
__global__ void convolutionGlobalMemory(float* output, const float* input, const float* filter, 
                                       int width, int height, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int halfFilter = filterSize / 2;
        
        for (int fy = -halfFilter; fy <= halfFilter; fy++) {
            for (int fx = -halfFilter; fx <= halfFilter; fx++) {
                int px = x + fx;
                int py = y + fy;
                
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int filterIdx = (fy + halfFilter) * filterSize + (fx + halfFilter);
                    sum += input[py * width + px] * filter[filterIdx];
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}

void runConstantMemoryExample(int size) {
    // Görüntü boyutları
    int width = 1024;
    int height = size / (width * sizeof(float));
    if (height < 512) height = 512;
    
    int imageSize = width * height;
    int filterSize = 5; // 5x5 filter
    int filterElements = filterSize * filterSize;
    
    // Host bellek
    float* h_input = new float[imageSize];
    float* h_output = new float[imageSize];
    float* h_filter = new float[filterElements];
    
    // Filter'ı başlat (Gaussian blur)
    float sigma = 1.0f;
    float sum = 0.0f;
    int halfFilter = filterSize / 2;
    
    for (int y = -halfFilter; y <= halfFilter; y++) {
        for (int x = -halfFilter; x <= halfFilter; x++) {
            int idx = (y + halfFilter) * filterSize + (x + halfFilter);
            h_filter[idx] = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            sum += h_filter[idx];
        }
    }
    
    // Normalize
    for (int i = 0; i < filterElements; i++) {
        h_filter[i] /= sum;
    }
    
    // Görüntüyü başlat
    for (int i = 0; i < imageSize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Device bellek
    float *d_input, *d_output, *d_filter;
    CUDA_CHECK(cudaMalloc(&d_input, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, filterElements * sizeof(float)));
    
    // Veriyi kopyala
    CUDA_CHECK(cudaMemcpy(d_input, h_input, imageSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filterElements * sizeof(float), cudaMemcpyHostToDevice));
    
    // Filter'ı constant memory'ye kopyala
    CUDA_CHECK(cudaMemcpyToSymbol(d_constData, h_filter, filterElements * sizeof(float)));
    
    // Kernel konfigürasyonu
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Constant memory versiyonu
    CUDA_CHECK(cudaEventRecord(start));
    convolutionConstantMemory<<<gridSize, blockSize>>>(d_output, d_input, width, height, filterSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float constantTime;
    CUDA_CHECK(cudaEventElapsedTime(&constantTime, start, stop));
    
    // Global memory versiyonu
    CUDA_CHECK(cudaEventRecord(start));
    convolutionGlobalMemory<<<gridSize, blockSize>>>(d_output, d_input, d_filter, width, height, filterSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float globalTime;
    CUDA_CHECK(cudaEventElapsedTime(&globalTime, start, stop));
    
    std::cout << "  Convolution (" << width << "x" << height << ", " << filterSize << "x" << filterSize << " filter):" << std::endl;
    std::cout << "    Constant memory time: " << constantTime << " ms" << std::endl;
    std::cout << "    Global memory time: " << globalTime << " ms" << std::endl;
    std::cout << "    Speedup: " << globalTime / constantTime << "x" << std::endl;
    
    // Belleği temizle
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_filter));
    delete[] h_input;
    delete[] h_output;
    delete[] h_filter;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}