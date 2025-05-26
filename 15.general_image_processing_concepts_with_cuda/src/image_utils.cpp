#include "image_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>

// ===== ImageIO Class Implementation =====

Image* ImageIO::loadPPM(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return nullptr;
    }
    
    std::string magic;
    file >> magic;
    
    if (magic != "P3" && magic != "P6") {
        std::cerr << "Error: Unsupported PPM format. Only P3 and P6 supported." << std::endl;
        return nullptr;
    }
    
    int width, height, maxval;
    file >> width >> height >> maxval;
    
    if (maxval != 255) {
        std::cerr << "Error: Only 8-bit PPM files supported (maxval=255)" << std::endl;
        return nullptr;
    }
    
    Image* image = new Image(width, height, 3);
    
    if (magic == "P3") {
        // ASCII format
        for (int i = 0; i < width * height * 3; i++) {
            int value;
            file >> value;
            image->data[i] = (unsigned char)value;
        }
    } else {
        // Binary format
        file.ignore(); // Skip whitespace after maxval
        file.read(reinterpret_cast<char*>(image->data), width * height * 3);
    }
    
    file.close();
    return image;
}

bool ImageIO::savePPM(const std::string& filename, const Image& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "P6\n";
    file << image.width << " " << image.height << "\n";
    file << "255\n";
    
    // Write binary data
    file.write(reinterpret_cast<const char*>(image.data), 
               image.width * image.height * image.channels);
    
    file.close();
    return true;
}

bool ImageIO::saveGrayscalePPM(const std::string& filename, const Image& image) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return false;
    }
    
    // Write header for grayscale
    file << "P5\n";
    file << image.width << " " << image.height << "\n";
    file << "255\n";
    
    // Write binary data
    file.write(reinterpret_cast<const char*>(image.data), 
               image.width * image.height);
    
    file.close();
    return true;
}

Image* ImageIO::createSampleImage(int width, int height) {
    Image* image = new Image(width, height, 3);
    
    std::cout << "Creating sample " << width << "x" << height << " test image..." << std::endl;
    
    // Create a colorful test pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            // Create color gradients and patterns
            float fx = (float)x / width;
            float fy = (float)y / height;
            
            // Red channel: horizontal gradient
            image->data[idx] = (unsigned char)(255 * fx);
            
            // Green channel: vertical gradient
            image->data[idx + 1] = (unsigned char)(255 * fy);
            
            // Blue channel: checkerboard pattern
            int checkSize = 32;
            bool check = ((x / checkSize) + (y / checkSize)) % 2;
            image->data[idx + 2] = check ? 255 : 128;
            
            // Add some circular patterns
            float centerX = width / 2.0f;
            float centerY = height / 2.0f;
            float dist = sqrtf((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
            float maxDist = sqrtf(centerX * centerX + centerY * centerY);
            
            if (dist < maxDist * 0.3f) {
                // Central circle - modify colors
                image->data[idx] = std::min(255, (int)(image->data[idx] * 1.5f));
                image->data[idx + 1] = std::min(255, (int)(image->data[idx + 1] * 0.5f));
                image->data[idx + 2] = std::min(255, (int)(image->data[idx + 2] * 1.2f));
            }
        }
    }
    
    return image;
}

void ImageIO::printImageInfo(const Image& image) {
    std::cout << "Image Information:" << std::endl;
    std::cout << "  Dimensions: " << image.width << "x" << image.height << std::endl;
    std::cout << "  Channels: " << image.channels << std::endl;
    std::cout << "  Total pixels: " << (image.width * image.height) << std::endl;
    std::cout << "  Memory size: " << (image.width * image.height * image.channels) << " bytes" << std::endl;
}

bool ImageIO::validateImage(const Image& image) {
    if (image.width <= 0 || image.height <= 0) {
        std::cerr << "Error: Invalid image dimensions" << std::endl;
        return false;
    }
    
    if (image.channels != 1 && image.channels != 3) {
        std::cerr << "Error: Unsupported number of channels" << std::endl;
        return false;
    }
    
    if (!image.data) {
        std::cerr << "Error: Image data is null" << std::endl;
        return false;
    }
    
    return true;
}

// ===== Timer Class Implementation =====

Timer::Timer() : start_time(0.0), end_time(0.0) {}

void Timer::start() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    start_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

void Timer::stop() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    end_time = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

double Timer::getElapsedTime() const {
    return end_time - start_time;
}

// ===== CPUReference Class Implementation =====

void CPUReference::grayscaleCPU(const Image& input, Image& output) {
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            int input_idx = (y * input.width + x) * 3;
            int output_idx = y * input.width + x;
            
            // Grayscale conversion
            float gray = 0.299f * input.data[input_idx] + 
                        0.587f * input.data[input_idx + 1] + 
                        0.114f * input.data[input_idx + 2];
            
            output.data[output_idx] = (unsigned char)std::min(255.0f, std::max(0.0f, gray));
        }
    }
}

void CPUReference::gaussianBlurCPU(const Image& input, Image& output, float sigma) {
    // Generate Gaussian kernel
    int kernelSize = (int)(6 * sigma) | 1; // Ensure odd size
    if (kernelSize > 15) kernelSize = 15;
    
    float* kernel = new float[kernelSize * kernelSize];
    float sum = 0.0f;
    int half = kernelSize / 2;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + half) * kernelSize + (x + half)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Apply blur
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            for (int c = 0; c < input.channels; c++) {
                float blurValue = 0.0f;
                
                for (int ky = 0; ky < kernelSize; ky++) {
                    for (int kx = 0; kx < kernelSize; kx++) {
                        int iy = y + ky - half;
                        int ix = x + kx - half;
                        
                        // Handle border conditions (clamp to edge)
                        iy = std::max(0, std::min(input.height - 1, iy));
                        ix = std::max(0, std::min(input.width - 1, ix));
                        
                        int input_idx = (iy * input.width + ix) * input.channels + c;
                        float kernelValue = kernel[ky * kernelSize + kx];
                        blurValue += kernelValue * input.data[input_idx];
                    }
                }
                
                int output_idx = (y * input.width + x) * input.channels + c;
                output.data[output_idx] = (unsigned char)std::min(255.0f, std::max(0.0f, blurValue));
            }
        }
    }
    
    delete[] kernel;
}

void CPUReference::sobelCPU(const Image& input, Image& output) {
    // Sobel kernels
    float sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    for (int y = 1; y < input.height - 1; y++) {
        for (int x = 1; x < input.width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int input_idx = ((y + ky) * input.width + (x + kx)) * 3;
                    
                    // Convert to grayscale first
                    float gray = 0.299f * input.data[input_idx] + 
                                0.587f * input.data[input_idx + 1] + 
                                0.114f * input.data[input_idx + 2];
                    
                    int kernel_idx = (ky + 1) * 3 + (kx + 1);
                    gx += sobelX[kernel_idx] * gray;
                    gy += sobelY[kernel_idx] * gray;
                }
            }
            
            float magnitude = sqrtf(gx * gx + gy * gy);
            int output_idx = y * input.width + x;
            output.data[output_idx] = (unsigned char)std::min(255.0f, std::max(0.0f, magnitude));
        }
    }
    
    // Handle borders (set to zero)
    for (int x = 0; x < input.width; x++) {
        output.data[x] = 0; // Top row
        output.data[(input.height - 1) * input.width + x] = 0; // Bottom row
    }
    for (int y = 0; y < input.height; y++) {
        output.data[y * input.width] = 0; // Left column
        output.data[y * input.width + (input.width - 1)] = 0; // Right column
    }
}

void CPUReference::brightnessCPU(const Image& input, Image& output, float factor) {
    for (int i = 0; i < input.width * input.height * input.channels; i++) {
        float newValue = input.data[i] * factor;
        output.data[i] = (unsigned char)std::min(255.0f, std::max(0.0f, newValue));
    }
}