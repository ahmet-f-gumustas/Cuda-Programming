#include "memory_manager.cuh"

// Shared memory kullanan matrix transpose
template<int TILE_DIM, int BLOCK_ROWS>
__global__ void transposeWithSharedMemory(float* odata, const float* idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 bank conflict'i önlemek için
    
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + yIndex * width;
    
    // Shared memory'ye yükle
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && yIndex + i < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
        }
    }
    
    __syncthreads();
    
    // Transpoze edilmiş indeksler
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + yIndex * height;
    
    // Shared memory'den transpoze ederek yaz
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < height && yIndex + i < width) {
            odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

// Shared memory kullanmayan versiyon (karşılaştırma için)
__global__ void transposeNaive(float* odata, const float* idata, int width, int height) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (xIndex < width && yIndex < height) {
        int index_in = xIndex + yIndex * width;
        int index_out = yIndex + xIndex * height;
        odata[index_out] = idata[index_in];
    }
}

// 1D reduction with shared memory
__global__ void reduceWithSharedMemory(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Her thread 2 element yükler
    float mySum = 0;
    if (idx < n) mySum += g_idata[idx];
    if (idx + blockDim.x < n) mySum += g_idata[idx + blockDim.x];
    
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // İlk thread sonucu yazar
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void runSharedMemoryExample(int size) {
    // Matrix transpose örneği
    int width = 1024;
    int height = size / width;
    if (height < 1) height = 1;
    
    size_t matrixSize = width * height * sizeof(float);
    
    float* h_idata = new float[width * height];
    float* h_odata = new float[width * height];
    float* d_idata, *d_odata;
    
    // Matrisi başlat
    for (int i = 0; i < width * height; i++) {
        h_idata[i] = static_cast<float>(i);
    }
    
    CUDA_CHECK(cudaMalloc(&d_idata, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_odata, matrixSize));
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, matrixSize, cudaMemcpyHostToDevice));
    
    // Kernel parametreleri
    const int TILE_DIM = 32;
    const int BLOCK_ROWS = 8;
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    
    // Shared memory kullanan versiyon
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    transposeWithSharedMemory<TILE_DIM, BLOCK_ROWS><<<dimGrid, dimBlock>>>(d_odata, d_idata, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float sharedTime;
    CUDA_CHECK(cudaEventElapsedTime(&sharedTime, start, stop));
    
    // Naive versiyon
    dim3 dimBlockNaive(16, 16);
    dim3 dimGridNaive((width + 15) / 16, (height + 15) / 16);
    
    CUDA_CHECK(cudaEventRecord(start));
    transposeNaive<<<dimGridNaive, dimBlockNaive>>>(d_odata, d_idata, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float naiveTime;
    CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
    
    std::cout << "  Matrix Transpose (" << width << "x" << height << "):" << std::endl;
    std::cout << "    Shared memory time: " << sharedTime << " ms" << std::endl;
    std::cout << "    Naive time: " << naiveTime << " ms" << std::endl;
    std::cout << "    Speedup: " << naiveTime / sharedTime << "x" << std::endl;
    
    // Belleği temizle
    CUDA_CHECK(cudaFree(d_idata));
    CUDA_CHECK(cudaFree(d_odata));
    delete[] h_idata;
    delete[] h_odata;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}