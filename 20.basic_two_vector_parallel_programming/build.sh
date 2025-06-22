#!/bin/bash

# CUDA Vector Addition Build Script

echo "CUDA Vector Addition - CMake Build Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" 
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is installed
print_status "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    print_error "NVCC not found! Please install CUDA Toolkit."
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found! Please check NVIDIA driver installation."
    exit 1
fi

print_success "CUDA installation found"
nvcc --version
echo ""

# Check if CMake is installed
print_status "Checking CMake installation..."
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found! Please install CMake (version 3.18+)."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_success "CMake $CMAKE_VERSION found"
echo ""

# Create project directory structure
print_status "Creating project structure..."
mkdir -p src
mkdir -p build

# Create source files if they don't exist
if [ ! -f "CMakeLists.txt" ]; then
    print_warning "CMakeLists.txt not found! Please make sure all files are in place."
fi

if [ ! -f "src/vector_add.cu" ]; then
    print_warning "src/vector_add.cu not found! Please make sure all files are in place."
fi

# Build the project
print_status "Building project..."
cd build

# Configure with CMake
print_status "Configuring with CMake..."
if cmake .. -DCMAKE_BUILD_TYPE=Release; then
    print_success "CMake configuration successful"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build
print_status "Compiling CUDA code..."
if make -j$(nproc); then
    print_success "Build successful"
else
    print_error "Build failed"
    exit 1
fi

echo ""
print_success "Project built successfully!"
echo ""
print_status "Available executables:"
ls -la vector_add* 2>/dev/null || echo "No executables found"

echo ""
print_status "To run the programs:"
echo "  ./vector_add              - Basic version"
echo "  ./vector_add_advanced     - Advanced version with performance analysis"
echo ""
print_status "Or use make targets:"
echo "  make run                  - Run basic version"
echo "  make run_advanced         - Run advanced version"
echo "  make cuda_info           - Show CUDA system information"

# Optional: Run basic test
echo ""
read -p "Do you want to run the basic version now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running basic vector addition..."
    if [ -f "./vector_add" ]; then
        ./vector_add
        print_success "Test completed successfully!"
    else
        print_error "Executable not found"
    fi
fi