#include "../include/common.h"
#include <cublas_v2.h>

// Naive matrix multiplication kernel
__global__ void matrix_multiply_naive_kernel(const float* A, const float* B, 
                                            float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared memory tiled matrix multiplication
#define TILE_SIZE 16

__global__ void matrix_multiply_shared_kernel(const float* A, const float* B, 
                                             float* C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            tile_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            tile_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Advanced shared memory with bank conflict avoidance
#define TILE_SIZE_ADV 32
#define TILE_SIZE_ADV_PADDED (TILE_SIZE_ADV + 1)

__global__ void matrix_multiply_advanced_kernel(const float* A, const float* B, 
                                               float* C, int N) {
    __shared__ float tile_A[TILE_SIZE_ADV][TILE_SIZE_ADV_PADDED];
    __shared__ float tile_B[TILE_SIZE_ADV][TILE_SIZE_ADV_PADDED];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE_ADV + ty;
    int col = bx * TILE_SIZE_ADV + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE_ADV - 1) / TILE_SIZE_ADV; ++t) {
        // Coalesced loading into shared memory
        if (row < N && t * TILE_SIZE_ADV + tx < N) {
            tile_A[ty][tx] = A[row * N + t * TILE_SIZE_ADV + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE_ADV + ty < N) {
            tile_B[ty][tx] = B[(t * TILE_SIZE_ADV + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unrolled inner loop for better performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_ADV; ++k) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result with coalesced access
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Warp-level matrix multiplication using tensor cores (conceptual)
__global__ void matrix_multiply_warp_kernel(const float* A, const float* B, 
                                           float* C, int N) {
    // This is a simplified version - real tensor core usage requires half precision
    // and more complex setup
    
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Each warp handles a 32x32 tile (or appropriate size)
    int warp_row = (warp_id / ((N + 31) / 32)) * 32;
    int warp_col = (warp_id % ((N + 31) / 32)) * 32;
    
    int row = warp_row + lane_id;
    int col = warp_col;
    
    if (row < N) {
        for (int c = 0; c < 32 && warp_col + c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + (warp_col + c)];
            }
            if (warp_col + c < N) {
                C[row * N + (warp_col + c)] = sum;
            }
        }
    }
}

// Naive implementation
void matrix_multiply_naive(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<float>& C, int N) {
    C.resize(N * N);
    
    ManagedMemory<float> d_A(N * N);
    ManagedMemory<float> d_B(N * N);
    ManagedMemory<float> d_C(N * N);
    
    d_A.copy_from_host(A.data());
    d_B.copy_from_host(B.data());
    
    CudaTimer timer;
    timer.start();
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    matrix_multiply_naive_kernel<<<grid, block>>>(d_A.get(), d_B.get(), d_C.get(), N);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_C.copy_to_host(C.data());
    
    // Calculate GFLOPS
    double operations = 2.0 * N * N * N; // 2*N^3 operations
    double gflops = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "Naive Matrix Multiplication - Time: " << timer.elapsed_ms() 
              << " ms, GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

// Shared memory implementation
void matrix_multiply_shared(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int N) {
    C.resize(N * N);
    
    ManagedMemory<float> d_A(N * N);
    ManagedMemory<float> d_B(N * N);
    ManagedMemory<float> d_C(N * N);
    
    d_A.copy_from_host(A.data());
    d_B.copy_from_host(B.data());
    
    CudaTimer timer;
    timer.start();
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matrix_multiply_shared_kernel<<<grid, block>>>(d_A.get(), d_B.get(), d_C.get(), N);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_C.copy_to_host(C.data());
    
    // Calculate GFLOPS
    double operations = 2.0 * N * N * N;
    double gflops = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "Shared Memory Matrix Multiplication - Time: " << timer.elapsed_ms() 
              << " ms, GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

// Advanced shared memory implementation
void matrix_multiply_advanced(const std::vector<float>& A, const std::vector<float>& B,
                             std::vector<float>& C, int N) {
    C.resize(N * N);
    
    ManagedMemory<float> d_A(N * N);
    ManagedMemory<float> d_B(N * N);
    ManagedMemory<float> d_C(N * N);
    
    d_A.copy_from_host(A.data());
    d_B.copy_from_host(B.data());
    
    CudaTimer timer;
    timer.start();
    
    dim3 block(TILE_SIZE_ADV, TILE_SIZE_ADV);
    dim3 grid((N + TILE_SIZE_ADV - 1) / TILE_SIZE_ADV, (N + TILE_SIZE_ADV - 1) / TILE_SIZE_ADV);
    
    matrix_multiply_advanced_kernel<<<grid, block>>>(d_A.get(), d_B.get(), d_C.get(), N);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_C.copy_to_host(C.data());
    
    // Calculate GFLOPS
    double operations = 2.0 * N * N * N;
    double gflops = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "Advanced Shared Memory Matrix Multiplication - Time: " << timer.elapsed_ms() 
              << " ms, GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

// cuBLAS implementation
void matrix_multiply_cublas(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int N) {
    C.resize(N * N);
    
    ManagedMemory<float> d_A(N * N);
    ManagedMemory<float> d_B(N * N);
    ManagedMemory<float> d_C(N * N);
    
    d_A.copy_from_host(A.data());
    d_B.copy_from_host(B.data());
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CudaTimer timer;
    timer.start();
    
    // cuBLAS uses column-major order, so we need to transpose
    // C = A * B becomes C^T = B^T * A^T
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B.get(), N,
                d_A.get(), N,
                &beta,
                d_C.get(), N);
    
    CUDA_CHECK_KERNEL();
    timer.stop();
    
    d_C.copy_to_host(C.data());
    
    // Calculate GFLOPS
    double operations = 2.0 * N * N * N;
    double gflops = operations / (timer.elapsed_ms() / 1000.0) / 1e9;
    
    std::cout << "cuBLAS Matrix Multiplication - Time: " << timer.elapsed_ms() 
              << " ms, GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    
    cublasDestroy(handle);
}

// CPU reference implementation
void matrix_multiply_cpu(const std::vector<float>& A, const std::vector<float>& B,
                        std::vector<float>& C, int N) {
    C.resize(N * N);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Calculate GFLOPS
    double operations = 2.0 * N * N * N;
    double gflops = operations / (cpu_time / 1000.0) / 1e9;
    
    std::cout << "CPU Matrix Multiplication - Time: " << cpu_time 
              << " ms, GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
}

// Test function
void test_matrix_multiply() {
    std::cout << "\n=== MATRIX MULTIPLICATION TEST ===" << std::endl;
    
    const int N = 1024; // 1024x1024 matrices
    
    // Generate random matrices
    auto A = generate_random_data<float>(N * N, 0.0f, 1.0f);
    auto B = generate_random_data<float>(N * N, 0.0f, 1.0f);
    
    std::vector<float> C_cpu, C_naive, C_shared, C_advanced, C_cublas;
    
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    
    // CPU reference (only for smaller matrices due to time)
    if (N <= 512) {
        matrix_multiply_cpu(A, B, C_cpu, N);
    }
    
    // GPU implementations
    matrix_multiply_naive(A, B, C_naive, N);
    matrix_multiply_shared(A, B, C_shared, N);
    matrix_multiply_advanced(A, B, C_advanced, N);
    matrix_multiply_cublas(A, B, C_cublas, N);
    
    // Verify correctness (compare with cuBLAS as reference)
    if (N <= 512 && !C_cpu.empty()) {
        // Compare with CPU
        bool naive_correct = true;
        bool shared_correct = true;
        bool advanced_correct = true;
        
        const float tolerance = 1e-3f;
        
        for (int i = 0; i < N * N && (naive_correct || shared_correct || advanced_correct); ++i) {
            if (naive_correct && std::abs(C_naive[i] - C_cpu[i]) > tolerance) {
                naive_correct = false;
            }
            if (shared_correct && std::abs(C_shared[i] - C_cpu[i]) > tolerance) {
                shared_correct = false;
            }
            if (advanced_correct && std::abs(C_advanced[i] - C_cpu[i]) > tolerance) {
                advanced_correct = false;
            }
        }
        
        std::cout << "\nCorrectness (vs CPU):" << std::endl;
        std::cout << "Naive implementation: " << (naive_correct ? "✓" : "✗") << std::endl;
        std::cout << "Shared memory implementation: " << (shared_correct ? "✓" : "✗") << std::endl;
        std::cout << "Advanced implementation: " << (advanced_correct ? "✓" : "✗") << std::endl;
    }
    
    // Compare with cuBLAS
    bool shared_vs_cublas = true;
    bool advanced_vs_cublas = true;
    
    const float tolerance = 1e-2f; // Slightly relaxed tolerance for cuBLAS comparison
    
    for (int i = 0; i < std::min(1000, N * N); ++i) { // Check first 1000 elements
        if (shared_vs_cublas && std::abs(C_shared[i] - C_cublas[i]) > tolerance) {
            shared_vs_cublas = false;
        }
        if (advanced_vs_cublas && std::abs(C_advanced[i] - C_cublas[i]) > tolerance) {
            advanced_vs_cublas = false;
        }
    }
    
    std::cout << "\nCorrectness (vs cuBLAS):" << std::endl;
    std::cout << "Shared memory implementation: " << (shared_vs_cublas ? "✓" : "✗") << std::endl;
    std::cout << "Advanced implementation: " << (advanced_vs_cublas ? "✓" : "✗") << std::endl;
    
    // Performance summary
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "Matrix multiplication demonstrates the importance of memory access patterns" << std::endl;
    std::cout << "and the effectiveness of shared memory for data reuse." << std::endl;
    
    // Show sample results (small submatrix)
    if (N >= 4) {
        std::cout << "\nSample 4x4 submatrix from result (top-left corner):" << std::endl;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                std::cout << std::fixed << std::setprecision(2) << C_cublas[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}