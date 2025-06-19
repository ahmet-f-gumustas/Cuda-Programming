#!/bin/bash

# GStreamer CUDA PointCloud Processor Build Script

echo "=== GStreamer CUDA PointCloud Processor Build Script ==="

# Gerekli paketleri kontrol et
echo "Gerekli paketler kontrol ediliyor..."

# GStreamer development paketleri
REQUIRED_PACKAGES=(
    "libgstreamer1.0-dev"
    "libgstreamer-plugins-base1.0-dev"
    "libgstreamer-plugins-good1.0-dev"
    "libgstreamer-plugins-bad1.0-dev"
    "gstreamer1.0-plugins-ugly"
    "gstreamer1.0-libav"
    "cmake"
    "build-essential"
    "pkg-config"
)

# Paket kontrolü
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii  $package "; then
        echo "❌ $package yüklü değil!"
        echo "Lütfen şu komutu çalıştırın:"
        echo "sudo apt-get install ${REQUIRED_PACKAGES[*]}"
        exit 1
    else
        echo "✅ $package yüklü"
    fi
done

# CUDA kontrolü
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA yüklü değil veya PATH'te bulunamıyor!"
    exit 1
else
    echo "✅ CUDA yüklü: $(nvcc --version | grep "release" | awk '{print $6}')"
fi

# Build dizinini oluştur
echo ""
echo "Build dizini hazırlanıyor..."
if [ -d "build" ]; then
    echo "Mevcut build dizini temizleniyor..."
    rm -rf build
fi

mkdir -p build
cd build

# CMake konfigürasyon
echo ""
echo "CMake konfigürasyonu yapılıyor..."
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake konfigürasyonu başarısız!"
    exit 1
fi

# Build işlemi
echo ""
echo "Proje derleniyor..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build işlemi başarısız!"
    exit 1
fi

echo ""
echo "✅ Build işlemi başarılı!"
echo ""
echo "Programı çalıştırmak için:"
echo "  cd build"
echo "  ./pointcloud_processor"
echo ""
echo "Veya:"
echo "  ./build/pointcloud_processor"

# GStreamer plugins kontrol et
echo ""
echo "GStreamer plugins kontrol ediliyor..."
gst-inspect-1.0 appsrc > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ GStreamer appsrc plugin mevcut"
else
    echo "❌ GStreamer appsrc plugin bulunamadı!"
fi

gst-inspect-1.0 appsink > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ GStreamer appsink plugin mevcut"
else
    echo "❌ GStreamer appsink plugin bulunamadı!"
fi

echo ""
echo "=== Build işlemi tamamlandı! ==="