# CUDA Vector Addition Project

Bu proje, CUDA kullanarak iki vektörü paralel olarak toplayan bir program içerir. RTX 4070 GPU'nuz için optimize edilmiştir.

## 📁 Proje Yapısı

```
CudaVectorAddition/
├── CMakeLists.txt              # CMake build configuration
├── build.sh                    # Otomatik build script
├── README.md                   # Bu dosya
├── src/
│   ├── vector_add.cu          # Basit CUDA vektör toplama
│   └── vector_add_advanced.cu # Gelişmiş versiyon (performans analizi)
└── build/                     # Build output directory
```

## 🚀 Hızlı Başlangıç

### Gereksinimler
- CUDA Toolkit 12.4+ (sisteminizde mevcut)
- CMake 3.18+
- GCC/G++ compiler
- NVIDIA GPU (RTX 4070 tespit edildi)

### Build ve Çalıştırma

#### Otomatik Build (Önerilen)
```bash
chmod +x build.sh
./build.sh
```

#### Manuel Build
```bash
# Build directory oluştur
mkdir build && cd build

# CMake ile configure et
cmake .. -DCMAKE_BUILD_TYPE=Release

# Derle
make -j$(nproc)

# Çalıştır
./vector_add                # Basit versiyon
./vector_add_advanced       # Gelişmiş versiyon
```

#### CMake Targets
```bash
# Build directory içinde
make run                    # Basit versiyonu çalıştır
make run_advanced          # Gelişmiş versiyonu çalıştır
make cuda_info             # CUDA sistem bilgilerini göster
```

## 🔧 Program Özellikleri

### Basit Versiyon (`vector_add.cu`)
- **Vektör boyutu**: 1024 eleman
- **Fonksiyonalite**: Temel vektör toplama
- **Çıktı**: İlk 10 sonuç + durum bilgisi

### Gelişmiş Versiyon (`vector_add_advanced.cu`)
- **Vektör boyutu**: 2048 eleman
- **Performans ölçümü**: GPU vs CPU karşılaştırması
- **Doğrulama**: CPU ile sonuç kontrol
- **Detaylı çıktı**: Throughput, hızlanma oranı, bellek kullanımı

## 💡 CUDA Kavramları

Bu projede kullanılan temel CUDA yapıları:

### Kernel Fonksiyonu
```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
    __syncthreads();
}
```

### Thread Indexing
- `blockIdx.x`: Block numarası
- `threadIdx.x`: Block içindeki thread numarası  
- `blockDim.x`: Block başına thread sayısı

### Bellek Yönetimi
- `cudaMalloc()`: GPU bellek ayırma
- `cudaMemcpy()`: Host-Device veri transferi
- `cudaFree()`: GPU bellek serbest bırakma

### Senkronizasyon
- `__syncthreads()`: Block içi thread senkronizasyonu
- `cudaDeviceSynchronize()`: Host-Device senkronizasyonu

## 📊 Performans Bilgileri

RTX 4070 için beklenen performans:
- **GPU Çekirdek Sayısı**: 5888 CUDA cores
- **Bellek Bant Genişliği**: 504.2 GB/s
- **Beklenen Hızlanma**: 10-50x (vektör boyutuna bağlı)

## 🛠️ CMake Konfigürasyonu

### GPU Mimarisi
```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4070 için Ada Lovelace
```

### Compiler Flags
```cmake
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
```

## 🐛 Hata Giderme

### Yaygın Hatalar

1. **"nvcc: command not found"**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **CMake CUDA bulamıyor**
   ```bash
   sudo apt update
   sudo apt install cmake nvidia-cuda-toolkit
   ```

3. **Architecture hatası**
   - CMakeLists.txt içinde `CMAKE_CUDA_ARCHITECTURES` değerini kontrol edin
   - RTX 4070 için `89` kullanın

### Debug Mode
```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
gdb ./vector_add
```

## 📚 Daha Fazla Bilgi

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
- [NVIDIA Developer Documentation](https://developer.nvidia.com/cuda-zone)

## 🎯 Sonraki Adımlar

1. **Shared Memory kullanımı**: Block içi veri paylaşımı
2. **Streams**: Asenkron işlemler
3. **cuBLAS**: Optimized linear algebra
4. **Profiling**: nvprof/Nsight ile performans analizi

---

**Not**: Bu proje CUDA programlama öğrenimi için tasarlanmıştır. Üretim ortamında optimizasyonlar gerekebilir.