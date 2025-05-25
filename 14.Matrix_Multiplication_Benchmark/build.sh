#!/bin/bash

# CUDA Matrix Multiplication Benchmark Build Script
# Bu script projeyi otomatik olarak derler

set -e  # Hata durumunda çık

echo "=== CUDA Matrix Multiplication Benchmark Build Script ==="
echo

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonksiyonlar
print_info() {
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

# CUDA kurulumunu kontrol et
check_cuda() {
    print_info "CUDA kurulumu kontrol ediliyor..."
    
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc bulunamadı! CUDA Toolkit kurulu değil."
        exit 1
    fi
    
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "CUDA ${CUDA_VERSION} bulundu"
    
    # GPU kontrolü
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi bulunamadı. GPU durumu kontrol edilemiyor."
    else
        print_info "GPU bilgileri:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits | \
        while IFS=, read -r name driver memory; do
            echo "  GPU: $name"
            echo "  Driver: $driver"
            echo "  Memory: ${memory} MB"
        done
    fi
}

# CMake kurulumunu kontrol et
check_cmake() {
    print_info "CMake kurulumu kontrol ediliyor..."
    
    if ! command -v cmake &> /dev/null; then
        print_error "CMake bulunamadı! Lütfen CMake kurun."
        exit 1
    fi
    
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_success "CMake ${CMAKE_VERSION} bulundu"
    
    # Minimum versiyon kontrolü
    CMAKE_MAJOR=$(echo $CMAKE_VERSION | cut -d'.' -f1)
    CMAKE_MINOR=$(echo $CMAKE_VERSION | cut -d'.' -f2)
    
    if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]); then
        print_error "CMake 3.18+ gerekli, mevcut: ${CMAKE_VERSION}"
        exit 1
    fi
}

# Build dizinini hazırla
prepare_build_dir() {
    print_info "Build dizini hazırlanıyor..."
    
    # Eğer build dizini varsa ve clean parametresi verilmişse temizle
    if [ "$1" == "clean" ] && [ -d "build" ]; then
        print_info "Mevcut build dizini temizleniyor..."
        rm -rf build
    fi
    
    if [ ! -d "build" ]; then
        mkdir build
        print_success "Build dizini oluşturuldu"
    else
        print_info "Mevcut build dizini kullanılıyor"
    fi
}

# CMake yapılandırması
configure_cmake() {
    print_info "CMake yapılandırması çalıştırılıyor..."
    
    cd build
    
    # Build type belirle
    BUILD_TYPE=${BUILD_TYPE:-Release}
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_CUDA_ARCHITECTURES="75;86;87;89" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    if [ $? -eq 0 ]; then
        print_success "CMake yapılandırması tamamlandı"
    else
        print_error "CMake yapılandırması başarısız!"
        cd ..
        exit 1
    fi
    
    cd ..
}

# Derleme işlemi
build_project() {
    print_info "Proje derleniyor..."
    
    cd build
    
    # CPU sayısını belirle
    JOBS=$(nproc)
    print_info "Paralel derleme: ${JOBS} thread"
    
    make -j$JOBS
    
    if [ $? -eq 0 ]; then
        print_success "Derleme başarıyla tamamlandı!"
    else
        print_error "Derleme başarısız!"
        cd ..
        exit 1
    fi
    
    cd ..
}

# Test çalıştırma
run_test() {
    print_info "Hızlı test çalıştırılıyor..."
    
    cd build
    
    # Küçük bir test çalıştır
    ./matrix_benchmark 64
    
    if [ $? -eq 0 ]; then
        print_success "Test başarıyla tamamlandı!"
    else
        print_warning "Test başarısız oldu veya hata oluştu"
    fi
    
    cd ..
}

# Yardım mesajı
show_help() {
    echo "Kullanım: $0 [OPTIONS]"
    echo
    echo "OPTIONS:"
    echo "  clean         Build dizinini temizleyip yeniden oluştur"
    echo "  debug         Debug modunda derle"
    echo "  test          Derleme sonrası hızlı test çalıştır"
    echo "  help          Bu yardım mesajını göster"
    echo
    echo "Örnekler:"
    echo "  $0              # Normal derleme"
    echo "  $0 clean        # Temiz derleme"
    echo "  $0 debug        # Debug modunda derleme"
    echo "  $0 clean test   # Temiz derleme + test"
    echo
}

# Ana fonksiyon
main() {
    # Parametreleri işle
    CLEAN_BUILD=false
    DEBUG_BUILD=false
    RUN_TEST=false
    
    for arg in "$@"; do
        case $arg in
            clean)
                CLEAN_BUILD=true
                ;;
            debug)
                DEBUG_BUILD=true
                export BUILD_TYPE=Debug
                ;;
            test)
                RUN_TEST=true
                ;;
            help|--help|-h)
                show_help
                exit 0
                ;;
            *)
                print_warning "Bilinmeyen parametre: $arg"
                ;;
        esac
    done
    
    # Ana işlemler
    check_cuda
    echo
    check_cmake
    echo
    
    if [ "$CLEAN_BUILD" = true ]; then
        prepare_build_dir clean
    else
        prepare_build_dir
    fi
    echo
    
    configure_cmake
    echo
    build_project
    echo
    
    if [ "$RUN_TEST" = true ]; then
        run_test
        echo
    fi
    
    print_success "Build işlemi tamamlandı!"
    print_info "Çalıştırmak için: ./build/matrix_benchmark"
    
    if [ "$DEBUG_BUILD" = true ]; then
        print_info "Debug modunda derlenmiştir. GDB ile debug edebilirsiniz:"
        print_info "  gdb ./build/matrix_benchmark"
    fi
}

# Script'i çalıştır
main "$@"