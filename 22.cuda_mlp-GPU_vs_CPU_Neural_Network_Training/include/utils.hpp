// include/utils.hpp

#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>

struct Dataset {
    std::vector<float> images;
    std::vector<float> labels;
    int num_samples;
    int image_size;
    int num_classes;
};

// MNIST data loader
Dataset load_mnist(const std::string& path, bool train = true);

// One-hot encoding
std::vector<float> one_hot_encode(int label, int num_classes);

// Accuracy calculation
float compute_accuracy(const std::vector<float>& predictions, 
                      const std::vector<float>& labels,
                      int num_samples, int num_classes);

// Random weight initialization
void xavier_init(float* weights, int fan_in, int fan_out);

// Save timing results
void save_timing(const std::string& filename, 
                const std::string& stage,
                const std::string& device, 
                float ms);

#endif // UTILS_HPP