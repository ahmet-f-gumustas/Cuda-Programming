#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include "image_processing.h"
#include <string>

// PPM Image I/O functions
class ImageIO {
public:
    // Load PPM image (P3 format - ASCII RGB)
    static Image* loadPPM(const std::string& filename);
    
    // Save PPM image
    static bool savePPM(const std::string& filename, const Image& image);
    
    // Save grayscale PPM image
    static bool saveGrayscalePPM(const std::string& filename, const Image& image);
    
    // Create sample image for testing
    static Image* createSampleImage(int width, int height);
    
    // Utility functions
    static void printImageInfo(const Image& image);
    static bool validateImage(const Image& image);
};

// Performance measurement utilities
class Timer {
private:
    double start_time;
    double end_time;
    
public:
    Timer();
    void start();
    void stop();
    double getElapsedTime() const; // returns time in milliseconds
};

// CPU reference implementations for comparison
class CPUReference {
public:
    static void grayscaleCPU(const Image& input, Image& output);
    static void gaussianBlurCPU(const Image& input, Image& output, float sigma);
    static void sobelCPU(const Image& input, Image& output);
    static void brightnessCPU(const Image& input, Image& output, float factor);
};

#endif // IMAGE_UTILS_H