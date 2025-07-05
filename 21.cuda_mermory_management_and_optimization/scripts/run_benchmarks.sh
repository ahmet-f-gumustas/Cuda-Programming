#!/bin/bash

# Proje dizinine git
cd "$(dirname "$0")/.."

# Build dizini oluştur
mkdir -p build
cd build

# CMake ile derle
echo "Building project..."
cmake ..
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Running benchmarks..."
    
    # Benchmarkları çalıştır
    ./memory_benchmark
    
    # Profiling için (opsiyonel)
    if command -v nvprof &> /dev/null; then
        echo -e "\n=== Running with nvprof ==="
        nvprof --print-gpu-summary ./memory_benchmark
    fi
    
    # Nsight Compute için (opsiyonel)
    if command -v ncu &> /dev/null; then
        echo -e "\n=== Profiling with Nsight Compute ==="
        ncu --set full --export profile_results ./memory_benchmark
    fi
else
    echo "Build failed!"
    exit 1
fi