#include "matrix_benchmark.h"
#include <iostream>

// Basit (naive) CUDA kernel
__global__ void matrix_multiply_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled (shared memory kullanan) CUDA kernel
#define TILE_SIZE 16

__global__ void matrix_multiply_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float result = 0.0f;
    
    // Tüm tile'ları işle
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Shared memory'ye veri yükle
        if (row < N && (tile * TILE_SIZE + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && (tile * TILE_SIZE + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Tile içinde çarpım işlemi
        for (int k = 0; k < TILE_SIZE; k++) {
            result += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Sonucu yaz
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

// Host fonksiyonları
void gpu_matrix_multiply_naive(const float* A, const float* B, float* C, int N) {
    // Block ve grid boyutları
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);
    
    // Kernel'ı çalıştır
    matrix_multiply_naive_kernel<<<gridSize, blockSize>>>(A, B, C, N);
    
    // Hataları kontrol et
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_matrix_multiply_tiled(const float* A, const float* B, float* C, int N) {
    // Block ve grid boyutları
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Kernel'ı çalıştır
    matrix_multiply_tiled_kernel<<<gridSize, blockSize>>>(A, B, C, N);
    
    // Hataları kontrol et
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void cublas_matrix_multiply(cublasHandle_t handle, const float* A, const float* B, float* C, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // cuBLAS column-major formatında çalışır, bu yüzden A ve B'yi yer değiştiriyoruz
    CUBLAS_CHECK(cublasSgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             B, N,  // B matrisi
                             A, N,  // A matrisi
                             &beta,
                             C, N)); // Sonuç matrisi
}