#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Image structure
struct Image {
    unsigned char* data;
    int width;
    int height;
    int channels;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data = new unsigned char[width * height * channels];
    }
    
    ~Image() {
        delete[] data;
    }
};

// CUDA Kernels (Device functions)
extern "C" {
    // Grayscale conversion
    void launchGrayscaleKernel(unsigned char* d_input, unsigned char* d_output,
                              int width, int height);
    
    // Gaussian blur
    void launchGaussianBlurKernel(unsigned char* d_input, unsigned char* d_output,
                                 int width, int height, float sigma = 1.0f);
    
    // Edge detection (Sobel)
    void launchSobelKernel(unsigned char* d_input, unsigned char* d_output,
                          int width, int height);
    
    // Brightness adjustment
    void launchBrightnessKernel(unsigned char* d_input, unsigned char* d_output,
                               int width, int height, float factor);
}

// Host functions
class ImageProcessor {
private:
    unsigned char* d_input;
    unsigned char* d_output;
    size_t input_size;
    size_t output_size;
    
public:
    ImageProcessor();
    ~ImageProcessor();
    
    // Memory management
    void allocateMemory(int width, int height, int input_channels, int output_channels);
    void freeMemory();
    
    // Image processing functions
    void processGrayscale(const Image& input, Image& output);
    void processGaussianBlur(const Image& input, Image& output, float sigma = 1.0f);
    void processSobel(const Image& input, Image& output);
    void processBrightness(const Image& input, Image& output, float factor);
    
    // Utility functions
    void printDeviceInfo();
    double benchmarkKernel(void (*kernel_func)(), int iterations = 10);
};

#endif // IMAGE_PROCESSING_H