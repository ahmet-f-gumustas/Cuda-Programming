// src/neural_net_cpu.cpp

#include "neural_net.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std::chrono;

NeuralNetCPU::NeuralNetCPU(const NetworkConfig& cfg) : config(cfg) {
    int batch = config.batch_size;
    
    // Allocate weights and biases
    w1.resize(config.input_size * config.hidden_size);
    w2.resize(config.hidden_size * config.output_size);
    b1.resize(config.hidden_size);
    b2.resize(config.output_size);
    
    // Allocate gradients
    dw1.resize(w1.size());
    dw2.resize(w2.size());
    db1.resize(b1.size());
    db2.resize(b2.size());
    
    // Allocate activations
    h1.resize(batch * config.hidden_size);
    a1.resize(batch * config.hidden_size);
    z2.resize(batch * config.output_size);
    
    // Allocate backward pass storage
    grad_h1.resize(h1.size());
    grad_a1.resize(a1.size());
    grad_z2.resize(z2.size());
    
    // Initialize weights
    xavier_init(w1.data(), config.input_size, config.hidden_size);
    xavier_init(w2.data(), config.hidden_size, config.output_size);
    std::fill(b1.begin(), b1.end(), 0.0f);
    std::fill(b2.begin(), b2.end(), 0.0f);
}

void NeuralNetCPU::forward(const float* input, float* output) {
    auto start = high_resolution_clock::now();
    int batch = config.batch_size;
    
    // Layer 1: input -> hidden
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < config.hidden_size; ++j) {
            float sum = b1[j];
            for (int i = 0; i < config.input_size; ++i) {
                sum += input[b * config.input_size + i] * w1[i * config.hidden_size + j];
            }
            h1[b * config.hidden_size + j] = sum;
            // ReLU activation
            a1[b * config.hidden_size + j] = std::max(0.0f, sum);
        }
    }
    
    // Layer 2: hidden -> output
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < config.output_size; ++j) {
            float sum = b2[j];
            for (int i = 0; i < config.hidden_size; ++i) {
                sum += a1[b * config.hidden_size + i] * w2[i * config.output_size + j];
            }
            z2[b * config.output_size + j] = sum;
        }
    }
    
    // Softmax
    for (int b = 0; b < batch; ++b) {
        float max_val = *std::max_element(
            z2.begin() + b * config.output_size,
            z2.begin() + (b + 1) * config.output_size
        );
        
        float sum = 0.0f;
        for (int i = 0; i < config.output_size; ++i) {
            output[b * config.output_size + i] = 
                std::exp(z2[b * config.output_size + i] - max_val);
            sum += output[b * config.output_size + i];
        }
        
        for (int i = 0; i < config.output_size; ++i) {
            output[b * config.output_size + i] /= sum;
        }
    }
    
    auto end = high_resolution_clock::now();
    timing.forward_ms = duration<float, std::milli>(end - start).count();
}

void NeuralNetCPU::backward(const float* input, const float* target) {
    auto start = high_resolution_clock::now();
    int batch = config.batch_size;
    
    // Reset gradients
    std::fill(dw1.begin(), dw1.end(), 0.0f);
    std::fill(dw2.begin(), dw2.end(), 0.0f);
    std::fill(db1.begin(), db1.end(), 0.0f);
    std::fill(db2.begin(), db2.end(), 0.0f);
    
    // Compute output layer gradients (softmax + cross-entropy)
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < config.output_size; ++i) {
            // Gradient is (predicted - target) for softmax + CE loss
            grad_z2[b * config.output_size + i] = 
                z2[b * config.output_size + i] - target[b * config.output_size + i];
        }
    }
    
    // Backprop to hidden layer
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < config.hidden_size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < config.output_size; ++j) {
                sum += grad_z2[b * config.output_size + j] * 
                       w2[i * config.output_size + j];
            }
            // ReLU derivative
            grad_a1[b * config.hidden_size + i] = 
                (a1[b * config.hidden_size + i] > 0) ? sum : 0.0f;
        }
    }
    
    // Accumulate weight gradients
    for (int b = 0; b < batch; ++b) {
        // Layer 2 gradients
        for (int i = 0; i < config.hidden_size; ++i) {
            for (int j = 0; j < config.output_size; ++j) {
                dw2[i * config.output_size + j] += 
                    a1[b * config.hidden_size + i] * 
                    grad_z2[b * config.output_size + j];
            }
        }
        
        // Layer 1 gradients
        for (int i = 0; i < config.input_size; ++i) {
            for (int j = 0; j < config.hidden_size; ++j) {
                dw1[i * config.hidden_size + j] += 
                    input[b * config.input_size + i] * 
                    grad_a1[b * config.hidden_size + j];
            }
        }
        
        // Bias gradients
        for (int i = 0; i < config.output_size; ++i) {
            db2[i] += grad_z2[b * config.output_size + i];
        }
        for (int i = 0; i < config.hidden_size; ++i) {
            db1[i] += grad_a1[b * config.hidden_size + i];
        }
    }
    
    // Average gradients
    float scale = 1.0f / batch;
    for (auto& g : dw1) g *= scale;
    for (auto& g : dw2) g *= scale;
    for (auto& g : db1) g *= scale;
    for (auto& g : db2) g *= scale;
    
    auto end = high_resolution_clock::now();
    timing.backward_ms = duration<float, std::milli>(end - start).count();
}

void NeuralNetCPU::update_weights() {
    auto start = high_resolution_clock::now();
    
    // SGD update
    for (size_t i = 0; i < w1.size(); ++i) {
        w1[i] -= config.learning_rate * dw1[i];
    }
    for (size_t i = 0; i < w2.size(); ++i) {
        w2[i] -= config.learning_rate * dw2[i];
    }
    for (size_t i = 0; i < b1.size(); ++i) {
        b1[i] -= config.learning_rate * db1[i];
    }
    for (size_t i = 0; i < b2.size(); ++i) {
        b2[i] -= config.learning_rate * db2[i];
    }
    
    auto end = high_resolution_clock::now();
    timing.update_ms = duration<float, std::milli>(end - start).count();
}

float NeuralNetCPU::compute_loss(const float* output, const float* target) {
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