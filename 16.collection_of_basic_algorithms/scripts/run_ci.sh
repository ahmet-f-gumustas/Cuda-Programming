#!/bin/bash

# Continuous Integration Script for CUDA Parallel Algorithms Collection

set -e  # Exit on any error

echo "=== CUDA Parallel Algorithms Collection CI Pipeline ==="
echo "Starting continuous integration tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[CI]${NC} $1"
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

# Check prerequisites
print_status "Checking prerequisites..."

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA compiler (nvcc) not found"
    exit 1
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    print_error "No NVIDIA GPU detected or driver not installed"
    exit 1
fi

# Check CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found"
    exit 1
fi

print_success "All prerequisites satisfied"

# Build the project
print_status "Building project..."

# Clean previous build
if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCHITECTURES=89 \
         -DCMAKE_CUDA_FLAGS="-Xptxas -v --expt-extended-lambda"

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build
make -j$(nproc)

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

print_success "Build completed successfully"

# Run tests
print_status "Running test suite..."

./run_tests

if [ $? -ne 0 ]; then
    print_error "Tests failed"
    exit 1
fi

print_success "All tests passed"

# Run benchmarks
print_status "Running performance benchmarks..."

./run_benchmarks --iterations 5 --output benchmark_results_ci.txt

if [ $? -ne 0 ]; then
    print_warning "Benchmarks completed with warnings"
else
    print_success "Benchmarks completed successfully"
fi

# Performance regression check
print_status "Checking for performance regressions..."

if [ -f "../ci_baseline.txt" ]; then
    print_status "Comparing against baseline performance..."
    # Simple regression check (in a real CI, this would be more sophisticated)
    python3 ../scripts/analyze_results.py benchmark_results_ci.txt --report
else
    print_warning "No baseline found, saving current results as baseline"
    cp benchmark_results_ci.txt ../ci_baseline.txt
fi

# Memory leak check (using cuda-memcheck if available)
if command -v cuda-memcheck &> /dev/null; then
    print_status "Running memory leak detection..."
    cuda-memcheck --leak-check full ./parallel_algorithms --quick-test > memcheck.log 2>&1
    
    if grep -q "ERROR SUMMARY: 0 errors" memcheck.log; then
        print_success "No memory leaks detected"
    else
        print_warning "Potential memory issues detected, check memcheck.log"
    fi
else
    print_warning "cuda-memcheck not available, skipping memory leak detection"
fi

# Code coverage (if compiled with coverage flags)
if [ "$1" = "--coverage" ]; then
    print_status "Generating code coverage report..."
    # This would require compilation with coverage flags
    print_warning "Code coverage not implemented yet"
fi

# Generate final report
print_status "Generating CI report..."

cat > ci_report.txt << EOF
CUDA Parallel Algorithms Collection - CI Report
Generated: $(date)
Commit: $(git rev-parse --short HEAD 2>/dev/null || echo "Unknown")

BUILD STATUS: PASSED
TEST STATUS: PASSED
BENCHMARK STATUS: COMPLETED

GPU Information:
$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)

Performance Summary:
$(tail -n 10 benchmark_results_ci.txt)

All checks completed successfully!
EOF

print_success "CI pipeline completed successfully!"
print_status "Report saved to ci_report.txt"

# Return to original directory
cd ..

echo -e "${GREEN}=== CI PIPELINE COMPLETED ===${NC}"