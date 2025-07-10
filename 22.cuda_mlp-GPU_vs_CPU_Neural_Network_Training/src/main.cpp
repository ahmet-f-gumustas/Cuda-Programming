// src/main.cpp

#include "neural_net.hpp"
#include "utils.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

struct Args {
    std::string device = "gpu";
    int epochs = 10;
    int batch_size = 64;
    float learning_rate = 0.01f;
};

Args parse_args(int argc, char** argv) {
    Args args;
    
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (i + 1 < argc) {
            if (arg == "--device") {
                args.device = argv[i + 1];
            } else if (arg == "--epochs") {
                args.epochs = std::stoi(argv[i + 1]);
            } else if (arg == "--batch") {
                args.batch_size = std::stoi(argv[i + 1]);
            } else if (arg == "--lr") {
                args.learning_rate = std::stof(argv[i + 1]);
            }
        }
    }
    
    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    
    // Load dataset
    std::cout << "Loading MNIST dataset..." << std::endl;
    Dataset train_data = load_mnist("data/mnist", true);
    Dataset test_data = load_mnist("data/mnist", false);
    
    // Create network
    NetworkConfig config;
    config.batch_size = args.batch_size;
    config.learning_rate = args.learning_rate;
    
    std::unique_ptr<NeuralNet> net;
    if (args.device == "cpu") {
        net = std::make_unique<NeuralNetCPU>(config);
    } else {
        net = std::make_unique<NeuralNetGPU>(config);
    }
    
    std::cout << "Training on " << net->device_name() << " with:" << std::endl;
    std::cout << "  Epochs: " << args.epochs << std::endl;
    std::cout << "  Batch size: " << args.batch_size << std::endl;
    std::cout << "  Learning rate: " << args.learning_rate << std::endl;
    
    // Clear timing file
    std::ofstream timing_file("perf.csv");
    timing_file << "stage,device,ms" << std::endl;
    timing_file.close();
    
    // Training loop
    std::vector<float> batch_images(config.batch_size * config.input_size);
    std::vector<float> batch_labels(config.batch_size * config.output_size);
    std::vector<float> output(config.batch_size * config.output_size);
    
    int num_batches = train_data.num_samples / config.batch_size;
    
    for (int epoch = 0; epoch < args.epochs; ++epoch) {
        float total_loss = 0.0f;
        float total_forward_ms = 0.0f;
        float total_backward_ms = 0.0f;
        float total_update_ms = 0.0f;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            // Prepare batch
            int start_idx = batch * config.batch_size;
            std::copy(train_data.images.begin() + start_idx * config.input_size,
                     train_data.images.begin() + (start_idx + config.batch_size) * config.input_size,
                     batch_images.begin());
            std::copy(train_data.labels.begin() + start_idx * config.output_size,
                     train_data.labels.begin() + (start_idx + config.batch_size) * config.output_size,
                     batch_labels.begin());
            
            // Forward pass
            net->forward(batch_images.data(), output.data());
            
            // Compute loss
            float loss = net->compute_loss(output.data(), batch_labels.data());
            total_loss += loss;
            
            // Backward pass
            net->backward(batch_images.data(), batch_labels.data());
            
            // Update weights
            net->update_weights();
            
            // Accumulate timing
            TimingInfo timing = net->get_timing();
            total_forward_ms += timing.forward_ms;
            total_backward_ms += timing.backward_ms;
            total_update_ms += timing.update_ms;
        }
        
        // Save timing for this epoch
        save_timing("perf.csv", "forward", net->device_name(), total_forward_ms / num_batches);
        save_timing("perf.csv", "backward", net->device_name(), total_backward_ms / num_batches);
        save_timing("perf.csv", "update", net->device_name(), total_update_ms / num_batches);
        
        std::cout << "Epoch " << epoch + 1 << "/" << args.epochs 
                  << " - Loss: " << total_loss / num_batches
                  << " - Time: " << (total_forward_ms + total_backward_ms + total_update_ms) << " ms"
                  << std::endl;
    }
    
    // Test accuracy
    std::cout << "\nEvaluating on test set..." << std::endl;
    std::vector<float> all_predictions;
    int test_batches = test_data.num_samples / config.batch_size;
    
    for (int batch = 0; batch < test_batches; ++batch) {
        int start_idx = batch * config.batch_size;
        std::copy(test_data.images.begin() + start_idx * config.input_size,
                 test_data.images.begin() + (start_idx + config.batch_size) * config.input_size,
                 batch_images.begin());
        
        net->forward(batch_images.data(), output.data());
        all_predictions.insert(all_predictions.end(), output.begin(), output.end());
    }
    
    float accuracy = compute_accuracy(all_predictions, test_data.labels, 
                                     test_batches * config.batch_size, 
                                     config.output_size);
    
    std::cout << "Test accuracy: " << accuracy * 100 << "%" << std::endl;
    
    return 0;
}