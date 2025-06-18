#!/bin/bash

# CUDA Histogram Projesi Build Script

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== CUDA Histogram Projesi Build Script ===${NC}"

# Proje dizini kontrolü
PROJECT_NAME="18.cuda_histogram_project"

if [ ! -d "$PROJECT_NAME" ]; then
    echo -e "${YELLOW}Proje dizini oluşturuluyor: $PROJECT_NAME${NC}"
    mkdir -p $PROJECT_NAME
fi

cd $PROJECT_NAME

# Dizin yapısını oluştur
echo -e "${YELLOW}Dizin yapısı oluşturuluyor...${NC}"
mkdir -p include src build

echo -e "${GREEN}Dizin yapısı:${NC}"
echo "├── CMakeLists.txt"
echo "├── include/"
echo "│   ├── histogram_cpu.h"
echo "│   ├── histogram_cuda.h"
echo "│   └── utils.h"
echo "├── src/"
echo "│   ├── main.cpp"
echo "│   ├── histogram_cpu.cpp"
echo "│   ├── histogram_cuda.cu"
echo "│   └── utils.cpp"
echo "└── build/"

# CUDA kurulumu kontrolü
echo -e "${YELLOW}CUDA kurulumu kontrol ediliyor...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA derleyici bulundu${NC}"
    nvcc --version | head -n 1
else
    echo -e "${RED}✗ CUDA derleyici bulunamadı! Lütfen CUDA'yı kurun.${NC}"
    exit 1
fi

# GPU kontrolü
echo -e "${YELLOW}GPU kontrolü...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU bulundu${NC}"
    nvidia-smi -L
else
    echo -e "${RED}✗ NVIDIA GPU bulunamadı!${NC}"
    exit 1
fi

# CMake kontrolü
echo -e "${YELLOW}CMake kontrolü...${NC}"
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n 1)
    echo -e "${GREEN}✓ CMake bulundu: $CMAKE_VERSION${NC}"
    
    # Minimum versiyon kontrolü
    CMAKE_MIN_VERSION="3.18"
    CMAKE_CURRENT=$(cmake --version | grep -oP "(?<=cmake version )[0-9]+\.[0-9]+")
    
    if [ "$(printf '%s\n' "$CMAKE_MIN_VERSION" "$CMAKE_CURRENT" | sort -V | head -n1)" = "$CMAKE_MIN_VERSION" ]; then
        echo -e "${GREEN}✓ CMake versiyonu yeterli${NC}"
    else
        echo -e "${RED}✗ CMake versiyonu çok eski (minimum $CMAKE_MIN_VERSION gerekli)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ CMake bulunamadı! Lütfen CMake'i kurun.${NC}"
    exit 1
fi

# Build işlemi
echo -e "${YELLOW}Build işlemi başlatılıyor...${NC}"
cd build

# CMake konfigürasyonu
echo -e "${BLUE}CMake konfigürasyonu...${NC}"
if cmake .. -DCMAKE_BUILD_TYPE=Release; then
    echo -e "${GREEN}✓ CMake konfigürasyonu başarılı${NC}"
else
    echo -e "${RED}✗ CMake konfigürasyonu başarısız${NC}"
    exit 1
fi

# Derleme
echo -e "${BLUE}Derleme işlemi...${NC}"
if make -j$(nproc); then
    echo -e "${GREEN}✓ Derleme başarılı${NC}"
else
    echo -e "${RED}✗ Derleme başarısız${NC}"
    exit 1
fi

# Test çalıştırma
echo -e "${YELLOW}Program test ediliyor...${NC}"
if [ -f "./cuda_histogram" ]; then
    echo -e "${GREEN}✓ Executable oluşturuldu${NC}"
    echo -e "${BLUE}Program çalıştırılıyor...${NC}"
    ./cuda_histogram
else
    echo -e "${RED}✗ Executable bulunamadı${NC}"
    exit 1
fi

echo -e "${GREEN}=== Build işlemi tamamlandı! ===${NC}"
echo -e "${YELLOW}Projeyi çalıştırmak için:${NC}"
echo -e "cd $PROJECT_NAME/build"
echo -e "./cuda_histogram"