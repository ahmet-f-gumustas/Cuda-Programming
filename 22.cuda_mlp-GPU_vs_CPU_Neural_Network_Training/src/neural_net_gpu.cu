// src/neural_net_gpu.cu

#include "neural_net.hpp"
#include "utils.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// CUDA kernel for ReLU activation
__global__ void relu_forward(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// CUDA kernel for ReLU backward
__global__ void relu_backward(float* grad_input, const float* grad_output, 
                              const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0.0f;
    }
}

// CUDA kernel for softmax (stable version)
__global__ void softmax_forward(float* output, const float* input, 
                                int batch_size, int num_classes) {
    int batch = blockIdx.x;
    if (batch >= batch_size) return;
    
    int offset = batch * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (int i = 1; i < num_classes; ++i) {
        max_val = fmaxf(max_val, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        output[offset + i] = expf(input[offset + i] - max_val);
        sum += output[offset + i];
    }
    
    // Normalize
    for (int i = 0; i < num_classes; ++i) {
        output[offset + i] /= sum;
    }
}

// CUDA kernel for cross-entropy loss gradient
__global__ void ce_loss_backward(float* grad, const float* output, 
                                 const float* target, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = output[idx] - target[idx];
    }
}

// CUDA kernel for SGD weight update
__global__ void sgd_update(float* weights, const float* gradients, 
                          float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

// CUDA kernel for bias gradient accumulation
__global__ void accumulate_bias_gradient(float* db, const float* grad_output,
                                        int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            sum += grad_output[b * output_size + idx];
        }
        db[idx] = sum / batch_size;
    }
}

NeuralNetGPU::NeuralNetGPU(const NetworkConfig& cfg) : config(cfg) {
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublas_handle = handle;
    
    int batch = config.batch_size;
    
    // Allocate device memory for weights and biases
    CUDA_CHECK(cudaMalloc(&d_w1, config.input_size * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, config.hidden_size * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, config.output_size * sizeof(float)));
    
    // Allocate gradients
    CUDA_CHECK(cudaMalloc(&d_dw1, config.input_size * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dw2, config.hidden_size * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, config.output_size * sizeof(float)));
    
    // Allocate activations
    CUDA_CHECK(cudaMalloc(&d_h1, batch * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a1, batch * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2, batch * config.output_size * sizeof(float)));
    
    // Allocate backward pass storage
    CUDA_CHECK(cudaMalloc(&d_grad_h1, batch * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_a1, batch * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_z2, batch * config.output_size * sizeof(float)));
    
    // Allocate input/output/target buffers
    CUDA_CHECK(cudaMalloc(&d_input, batch * config.input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, batch * config.output_size * sizeof(float)));
    
    // Initialize weights on host and copy
    std::vector<float> h_w1(config.input_size * config.hidden_size);
    std::vector<float> h_w2(config.hidden_size * config.output_size);
    xavier_init(h_w1.data(), config.input_size, config.hidden_size);
    xavier_init(h_w2.data(), config.hidden_size, config.output_size);
    
    CUDA_CHECK(cudaMemcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_b1, 0, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b2, 0, config.output_size * sizeof(float)));
}

NeuralNetGPU::~NeuralNetGPU() {
    cublasDestroy((cublasHandle_t)cublas_handle);
    
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_dw1));
    CUDA_CHECK(cudaFree(d_dw2));
    CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_h1));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_z2));
    CUDA_CHECK(cudaFree(d_grad_h1));
    CUDA_CHECK(cudaFree(d_grad_a1));
    CUDA_CHECK(cudaFree(d_grad_z2));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_target));
}

void NeuralNetGPU::forward(const float* input, float* output) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int batch = config.batch_size;
    cublasHandle_t handle = (cublasHandle_t)cublas_handle;
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, input, batch * config.input_size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Layer 1: input @ w1 + b1
    float alpha = 1.0f, beta = 0.0f;
    
    // d_h1 = d_input @ d_w1 + d_b1
    // Using cuBLAS for matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                config.hidden_size, batch, config.input_size,
                &alpha,
                d_w1, config.hidden_size,
                d_input, config.input_size,
                &beta,
                d_h1, config.hidden_size);
    
    // Add bias (broadcast)
    cublasSger(handle, config.hidden_size, batch,
               &alpha,
               d_b1, 1,
               (float*)&alpha, 0,  // dummy vector of ones
               d_h1, config.hidden_size);
    
    // ReLU activation
    int block_size = 256;
    int grid_size = (batch * config.hidden_size + block_size - 1) / block_size;
    relu_forward<<<grid_size, block_size>>>(d_a1, d_h1, batch * config.hidden_size);
    
    // Layer 2: hidden @ w2 + b2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                config.output_size, batch, config.hidden_size,
                &alpha,
                d_w2, config.output_size,
                d_a1, config.hidden_size,
                &beta,
                d_z2, config.output_size);
    
    // Add bias
    cublasSger(handle, config.output_size, batch,
               &alpha,
               d_b2, 1,
               (float*)&alpha, 0,
               d_z2, config.output_size);
    
    // Softmax
    softmax_forward<<<batch, 1>>>(d_output, d_z2, batch, config.output_size);
    
    // Copy output to host
    CUDA_CHECK(cudaMemcpy(output, d_output, batch * config.output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing.forward_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void NeuralNetGPU::backward(const float* input, const float* target) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int batch = config.batch_size;
    cublasHandle_t handle = (cublasHandle_t)cublas_handle;
    
    // Copy target to device
    CUDA_CHECK(cudaMemcpy(d_target, target, batch * config.output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Reset gradients
    CUDA_CHECK(cudaMemset(d_dw1, 0, config.input_size * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dw2, 0, config.hidden_size * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db1, 0, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_db2, 0, config.output_size * sizeof(float)));
    
    // Compute output gradient (CE loss with softmax)
    int block_size = 256;
    int grid_size = (batch * config.output_size + block_size - 1) / block_size;
    ce_loss_backward<<<grid_size, block_size>>>(d_grad_z2, d_output, d_target, 
                                                batch * config.output_size);
    
    // Backprop through layer 2
    float alpha = 1.0f / batch, beta = 0.0f;
    
    // d_dw2 = d_a1.T @ d_grad_z2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                config.output_size, config.hidden_size, batch,
                &alpha,
                d_grad_z2, config.output_size,
                d_a1, config.hidden_size,
                &beta,
                d_dw2, config.output_size);
    
    // d_grad_a1 = d_grad_z2 @ d_w2.T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                config.hidden_size, batch, config.output_size,
                &alpha,
                d_w2, config.output_size,
                d_grad_z2, config.output_size,
                &beta,
                d_grad_a1, config.hidden_size);
    
    // Bias gradient
    grid_size = (config.output_size + block_size - 1) / block_size;
    accumulate_bias_gradient<<<grid_size, block_size>>>(d_db2, d_grad_z2, 
                                                        batch, config.output_size);
    
    // Backprop through ReLU
    grid_size = (batch * config.hidden_size + block_size - 1) / block_size;
    relu_backward<<<grid_size, block_size>>>(d_grad_h1, d_grad_a1, d_h1, 
                                            batch * config.hidden_size);
    
    // Backprop through layer 1
    // d_dw1 = d_input.T @ d_grad_h1
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                config.hidden_size, config.input_size, batch,
                &alpha,
                d_grad_h1, config.hidden_size,
                d_input, config.input_size,
                &beta,
                d_dw1, config.hidden_size);
    
    // Bias gradient
    grid_size = (config.hidden_size + block_size - 1) / block_size;
    accumulate_bias_gradient<<<grid_size, block_size>>>(d_db1, d_grad_h1, 
                                                        batch, config.hidden_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing.backward_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void NeuralNetGPU::update_weights() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    int block_size = 256;
    
    // Update weights and biases using SGD
    int grid_size = (config.input_size * config.hidden_size + block_size - 1) / block_size;
    sgd_update<<<grid_size, block_size>>>(d_w1, d_dw1, config.learning_rate, 
                                          config.input_size * config.hidden_size);
    
    grid_size = (config.hidden_size * config.output_size + block_size - 1) / block_size;
    sgd_update<<<grid_size, block_size>>>(d_w2, d_dw2, config.learning_rate,
                                          config.hidden_size * config.output_size);
    
    grid_size = (config.hidden_size + block_size - 1) / block_size;
    sgd_update<<<grid_size, block_size>>>(d_b1, d_db1, config.learning_rate,
                                          config.hidden_size);
    
    grid_size = (config.output_size + block_size - 1) / block_size;
    sgd_update<<<grid_size, block_size>>>(d_b2, d_db2, config.learning_rate,
                                          config.output_size);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timing.update_ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

float NeuralNetGPU::compute_loss(const float* output, const float* target) {
    float loss = 0.0f;
    int batch = config.batch_size;
    
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < config.output_size; ++i) {
            if (target[b * config.output_size + i] > 0) {
                loss -= target[b * config.output_size + i] * 
                        std::log(output[b * config.output_size + i] + 1e-7f);
            }
        }
    }
    
    return loss / batch;
}