#!/bin/bash

# CUDA Image Processing Build Script
echo "=== Building CUDA Image Processing Project ==="

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
    echo "Created build directory"
fi

cd build

# Configure with CMake
echo "Configuring project with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed"
    exit 1
fi

# Build the project
echo "Building project..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo "Build successful!"

# Go back to project root
cd ..

# Create sample images if they don't exist
if [ ! -d "images" ]; then
    echo "Creating sample images..."
    python3 create_sample_image.py
fi

echo ""
echo "=== Build Complete ==="
echo "To run the program:"
echo "  cd build"
echo "  ./image_processor ../images/sample.ppm output.ppm grayscale"
echo ""
echo "Available filters:"
echo "  grayscale                    - Convert to grayscale"
echo "  blur [sigma]                 - Gaussian blur (default sigma=1.0)"
echo "  edge                         - Sobel edge detection"
echo "  brightness <factor>          - Adjust brightness"
echo ""
echo "Examples:"
echo "  ./image_processor ../images/sample.ppm output.ppm blur 2.0"
echo "  ./image_processor ../images/sample.ppm output.ppm brightness 1.5"