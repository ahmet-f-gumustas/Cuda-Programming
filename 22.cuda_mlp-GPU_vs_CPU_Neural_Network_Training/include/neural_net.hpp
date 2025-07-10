// include/neural_net.hpp

#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <string>

struct NetworkConfig {
    int input_size = 784;
    int hidden_size = 128;
    int output_size = 10;
    float learning_rate = 0.01f;
    int batch_size = 64;
};

struct TimingInfo {
    float forward_ms = 0.0f;
    float backward_ms = 0.0f;
    float update_ms = 0.0f;
};

class NeuralNet {
public:
    virtual ~NeuralNet() = default;
    
    virtual void forward(const float* input, float* output) = 0;
    virtual void backward(const float* input, const float* target) = 0;
    virtual void update_weights() = 0;
    virtual float compute_loss(const float* output, const float* target) = 0;
    virtual TimingInfo get_timing() const = 0;
    virtual std::string device_name() const = 0;
};

// CPU implementation
class NeuralNetCPU : public NeuralNet {
private:
    NetworkConfig config;
    std::vector<float> w1, w2, b1, b2;
    std::vector<float> dw1, dw2, db1, db2;
    std::vector<float> h1, a1, z2;
    std::vector<float> grad_h1, grad_a1, grad_z2;
    TimingInfo timing;
    
public:
    explicit NeuralNetCPU(const NetworkConfig& cfg);
    
    void forward(const float* input, float* output) override;
    void backward(const float* input, const float* target) override;
    void update_weights() override;
    float compute_loss(const float* output, const float* target) override;
    TimingInfo get_timing() const override { return timing; }
    std::string device_name() const override { return "cpu"; }
};

// GPU implementation
class NeuralNetGPU : public NeuralNet {
private:
    NetworkConfig config;
    float *d_w1, *d_w2, *d_b1, *d_b2;
    float *d_dw1, *d_dw2, *d_db1, *d_db2;
    float *d_h1, *d_a1, *d_z2;
    float *d_grad_h1, *d_grad_a1, *d_grad_z2;
    float *d_input, *d_output, *d_target;
    TimingInfo timing;
    void* cublas_handle;
    
public:
    explicit NeuralNetGPU(const NetworkConfig& cfg);
    ~NeuralNetGPU();
    
    void forward(const float* input, float* output) override;
    void backward(const float* input, const float* target) override;
    void update_weights() override;
    float compute_loss(const float* output, const float* target) override;
    TimingInfo get_timing() const override { return timing; }
    std::string device_name() const override { return "gpu"; }
};

#endif // NEURAL_NET_HPP