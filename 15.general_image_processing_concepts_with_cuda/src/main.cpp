#include <iostream>
#include <string>
#include <chrono>
#include "image_processing.h"
#include "image_utils.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <input.ppm> <output.ppm> <filter> [params]\n";
    std::cout << "Filters:\n";
    std::cout << "  grayscale - Convert to grayscale\n";
    std::cout << "  blur [sigma] - Gaussian blur (default sigma=1.0)\n";
    std::cout << "  edge - Sobel edge detection\n";
    std::cout << "  brightness <factor> - Adjust brightness (e.g., 1.5 for 50% brighter)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << programName << " input.ppm output.ppm grayscale\n";
    std::cout << "  " << programName << " input.ppm output.ppm blur 2.0\n";
    std::cout << "  " << programName << " input.ppm output.ppm brightness 1.5\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string filter = argv[3];
    
    std::cout << "=== CUDA Image Processing ===\n";
    
    // Initialize CUDA and print device info
    ImageProcessor processor;
    processor.printDeviceInfo();
    
    // Load input image
    std::cout << "\nLoading image: " << inputFile << "\n";
    Timer timer;
    timer.start();
    
    Image* input = ImageIO::loadPPM(inputFile);
    if (!input) {
        std::cerr << "Error: Could not load image " << inputFile << "\n";
        std::cerr << "Creating sample image for testing...\n";
        input = ImageIO::createSampleImage(512, 512);
    }
    
    timer.stop();
    std::cout << "Image loading time: " << timer.getElapsedTime() << " ms\n";
    
    ImageIO::printImageInfo(*input);
    
    // Process image based on filter type
    Image* output = nullptr;
    double gpuTime = 0.0;
    double cpuTime = 0.0;
    
    if (filter == "grayscale") {
        output = new Image(input->width, input->height, 1);
        
        // GPU processing
        timer.start();
        processor.processGrayscale(*input, *output);
        timer.stop();
        gpuTime = timer.getElapsedTime();
        
        // CPU reference for comparison
        Image cpuOutput(input->width, input->height, 1);
        timer.start();
        CPUReference::grayscaleCPU(*input, cpuOutput);
        timer.stop();
        cpuTime = timer.getElapsedTime();
        
        std::cout << "\nGrayscale Conversion Results:\n";
        
    } else if (filter == "blur") {
        float sigma = 1.0f;
        if (argc > 4) {
            sigma = std::stof(argv[4]);
        }
        
        output = new Image(input->width, input->height, input->channels);
        
        // GPU processing
        timer.start();
        processor.processGaussianBlur(*input, *output, sigma);
        timer.stop();
        gpuTime = timer.getElapsedTime();
        
        // CPU reference
        Image cpuOutput(input->width, input->height, input->channels);
        timer.start();
        CPUReference::gaussianBlurCPU(*input, cpuOutput, sigma);
        timer.stop();
        cpuTime = timer.getElapsedTime();
        
        std::cout << "\nGaussian Blur Results (sigma=" << sigma << "):\n";
        
    } else if (filter == "edge") {
        output = new Image(input->width, input->height, 1);
        
        // GPU processing
        timer.start();
        processor.processSobel(*input, *output);
        timer.stop();
        gpuTime = timer.getElapsedTime();
        
        // CPU reference
        Image cpuOutput(input->width, input->height, 1);
        timer.start();
        CPUReference::sobelCPU(*input, cpuOutput);
        timer.stop();
        cpuTime = timer.getElapsedTime();
        
        std::cout << "\nSobel Edge Detection Results:\n";
        
    } else if (filter == "brightness") {
        if (argc < 5) {
            std::cerr << "Error: Brightness factor required\n";
            printUsage(argv[0]);
            delete input;
            return 1;
        }
        
        float factor = std::stof(argv[4]);
        output = new Image(input->width, input->height, input->channels);
        
        // GPU processing
        timer.start();
        processor.processBrightness(*input, *output, factor);
        timer.stop();
        gpuTime = timer.getElapsedTime();
        
        // CPU reference
        Image cpuOutput(input->width, input->height, input->channels);
        timer.start();
        CPUReference::brightnessCPU(*input, cpuOutput, factor);
        timer.stop();
        cpuTime = timer.getElapsedTime();
        
        std::cout << "\nBrightness Adjustment Results (factor=" << factor << "):\n";
        
    } else {
        std::cerr << "Error: Unknown filter '" << filter << "'\n";
        printUsage(argv[0]);
        delete input;
        return 1;
    }
    
    // Print performance results
    std::cout << "GPU processing time: " << gpuTime << " ms\n";
    std::cout << "CPU reference time:  " << cpuTime << " ms\n";
    if (cpuTime > 0) {
        std::cout << "Speedup: " << (cpuTime / gpuTime) << "x\n";
    }
    
    // Save output image
    std::cout << "\nSaving result to: " << outputFile << "\n";
    timer.start();
    
    bool saveSuccess = false;
    if (output->channels == 1) {
        saveSuccess = ImageIO::saveGrayscalePPM(outputFile, *output);
    } else {
        saveSuccess = ImageIO::savePPM(outputFile, *output);
    }
    
    timer.stop();
    
    if (saveSuccess) {
        std::cout << "Image saved successfully!\n";
        std::cout << "Save time: " << timer.getElapsedTime() << " ms\n";
    } else {
        std::cerr << "Error: Could not save image\n";
    }
    
    // Cleanup
    delete input;
    delete output;
    
    std::cout << "\n=== Processing Complete ===\n";
    return 0;
}