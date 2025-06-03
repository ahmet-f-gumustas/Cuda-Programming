#!/bin/bash

# CUDA Parallel Algorithms Collection Build Script
# RTX 4070 Ti Super | CUDA 12.4

echo "=== CUDA Parallel Algorithms Collection Build Script ==="
echo "Building for RTX 4070 Ti Super (Ada Lovelace Architecture)"

# Renkli output için
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build directory oluştur
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${BLUE}Creating build directory...${NC}"
    mkdir -p $BUILD_DIR
fi

cd $BUILD_DIR

# CMake configuration
echo -e "${BLUE}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCHITECTURES=89 \
         -DCMAKE_CUDA_FLAGS="-Xptxas -v --expt-extended-lambda" \
         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

# Build
echo -e "${BLUE}Building project...${NC}"
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo -e "${YELLOW}Executables created:${NC}"
    echo -e "  - ${GREEN}parallel_algorithms${NC} (main program)"
    echo -e "  - ${GREEN}run_tests${NC} (test suite)"
    echo -e "  - ${GREEN}run_benchmarks${NC} (benchmark suite)"
    
    echo -e "\n${YELLOW}Available make targets:${NC}"
    echo -e "  - ${GREEN}make profile${NC} (run with nvprof)"
    echo -e "  - ${GREEN}make occupancy${NC} (occupancy analysis)"
    
    echo -e "\n${YELLOW}To run the main program:${NC}"
    echo -e "  ${GREEN}./parallel_algorithms${NC}"
    
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

cd ..