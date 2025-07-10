// src/utils.cpp

#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>

Dataset load_mnist(const std::string& path, bool train) {
    Dataset dataset;
    
    // Simplified MNIST loader - generates random data for demo
    // In production, implement proper MNIST file parsing
    dataset.num_samples = train ? 60000 : 10000;
    dataset.image_size = 784;
    dataset.num_classes = 10;
    
    dataset.images.resize(dataset.num_samples * dataset.image_size);
    dataset.labels.resize(dataset.num_samples * dataset.num_classes);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> img_dist(0.0, 1.0);
    std::uniform_int_distribution<> label_dist(0, 9);
    
    // Generate random normalized images
    for (int i = 0; i < dataset.num_samples; ++i) {
        for (int j = 0; j < dataset.image_size; ++j) {
            dataset.images[i * dataset.image_size + j] = img_dist(gen);
        }
        
        // One-hot encode labels
        int label = label_dist(gen);
        for (int j = 0; j < dataset.num_classes; ++j) {
            dataset.labels[i * dataset.num_classes + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    
    return dataset;
}

std::vector<float> one_hot_encode(int label, int num_classes) {
    std::vector<float> encoded(num_classes, 0.0f);
    encoded[label] = 1.0f;
    return encoded;
}

float compute_accuracy(const std::vector<float>& predictions, 
                      const std::vector<float>& labels,
                      int num_samples, int num_classes) {
    int correct = 0;
    
    for (int i = 0; i < num_samples; ++i) {
        int pred_class = 0;
        float max_prob = predictions[i * num_classes];
        
        for (int j = 1; j < num_classes; ++j) {
            if (predictions[i * num_classes + j] > max_prob) {
                max_prob = predictions[i * num_classes + j];
                pred_class = j;
            }
        }
        
        int true_class = 0;
        for (int j = 0; j < num_classes; ++j) {
            if (labels[i * num_classes + j] > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return static_cast<float>(correct) / num_samples;
}

void xavier_init(float* weights, int fan_in, int fan_out) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (fan_in + fan_out));
    std::normal_distribution<> dist(0.0, scale);
    
    for (int i = 0; i < fan_in * fan_out; ++i) {
        weights[i] = dist(gen);
    }
}

void save_timing(const std::string& filename, 
                const std::string& stage,
                const std::string& device, 
                float ms) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << stage << "," << device << "," << ms << std::endl;
        file.close();
    }
}