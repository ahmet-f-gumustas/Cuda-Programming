# CUDA Paralel Histogram Hesaplama Projesi

Bu proje, CUDA C++ kullanarak paralel histogram hesaplama algoritmasını göstermektedir. CPU ve GPU implementasyonlarını karşılaştırarak CUDA'nın performans avantajlarını ortaya koyar.

## 🎯 Proje Özellikleri

- **CPU Single Thread**: Geleneksel tek iş parçacıklı histogram hesaplama
- **CPU Multi-Thread**: OpenMP kullanarak çoklu iş parçacıklı CPU hesaplama
- **CUDA Basic**: Basit CUDA kernel implementasyonu
- **CUDA Optimized**: Shared memory kullanarak optimize edilmiş CUDA kernel

## 📋 Gereksinimler

### Sistem Gereksinimleri
- NVIDIA GPU (Compute Capability 3.5+)
- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 uyumlu derleyici (GCC 7+, MSVC 2019+)

### Kütüphaneler
- CUDA Runtime
- Standard C++ Library
- Threading support

## 🏗️ Proje Yapısı

```
18.cuda_histogram_project/
├── CMakeLists.txt          # CMake build konfigürasyonu
├── include/                # Header dosyaları
│   ├── histogram_cpu.h     # CPU histogram fonksiyonları
│   ├── histogram_cuda.h    # CUDA histogram fonksiyonları
│   └── utils.h             # Yardımcı fonksiyonlar
├── src/                    # Kaynak kodlar
│   ├── main.cpp            # Ana program
│   ├── histogram_cpu.cpp   # CPU implementasyonu
│   ├── histogram_cuda.cu   # CUDA implementasyonu
│   └── utils.cpp           # Yardımcı fonksiyonlar
├── build/                  # Build dosyaları
├── build.sh               # Otomatik build script
└── README.md              # Bu dosya
```

## 🚀 Kurulum ve Çalıştırma

### Hızlı Başlangıç

1. **Proje dosyalarını oluşturun**: Yukarıdaki tüm dosyaları kopyalayın
2. **Build script'i çalıştırın**:
   ```bash
   chmod +x build.sh
   ./build.sh
   ```

### Manuel Kurulum

```bash
# Proje dizini oluştur
mkdir 18.cuda_histogram_project
cd 18.cuda_histogram_project

# Dizin yapısı oluştur
mkdir -p include src build

# Dosyaları kopyalayın (yukarıdaki artifact'lar)

# Build işlemi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Çalıştır
./cuda_histogram
```

### Debug Modunda Build

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

## 📊 Program Çıktısı

Program çalıştırıldığında aşağıdaki bilgileri gösterir:

1. **GPU Bilgileri**: Kullanılan NVIDIA GPU'nun özellikleri
2. **Test Parametreleri**: Veri boyutu, bin sayısı, değer aralığı
3. **Veri İstatistikleri**: Oluşturulan test verilerinin özellikleri
4. **Performans Sonuçları**: Her algoritmanın çalışma süresi
5. **Hızlanma Oranları**: Algoritmaların birbirine göre performansı
6. **Doğrulama**: Sonuçların tutarlılığı kontrolü
7. **Histogram İstatistikleri**: Hesaplanan histogram özellikleri

### Örnek Çıktı

```
=== CUDA Paralel Histogram Hesaplama Projesi ===

=== GPU BİLGİLERİ ===
GPU sayısı: 1
GPU 0: NVIDIA GeForce RTX 4070 Laptop GPU
- Compute Capability: 8.9
- Global Memory: 8188 MB
- Shared Memory per Block: 48 KB
- Max Threads per Block: 1024

Test Parametreleri:
- Veri boyutu: 1000000
- Bin sayısı: 256
- Değer aralığı: [0, 255]

=== PERFORMANS SONUÇLARI ===
CPU (Single Thread):  125.4523 ms
CPU (Parallel):       18.7821 ms
CUDA (Basic):         2.3456 ms
CUDA (Optimized):     1.1234 ms

=== HIZLANMA ORANLARI ===
CPU Parallel, CPU Single'dan 6.68x daha hızlı
CUDA Basic, CPU Single'dan 53.48x daha hızlı
CUDA Optimized, CPU Single'dan 111.67x daha hızlı
```

## 🔧 Konfigürasyon

### GPU Mimarisi Ayarları

RTX 4070 için CMakeLists.txt'de:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

Diğer GPU'lar için:
- RTX 3090/4090: `86`
- GTX 1080: `61`
- Tesla V100: `70`

### Kernel Parametreleri

`histogram_cuda.cu` dosyasında ayarlanabilir:
```cpp
int block_size = 256;           // Thread sayısı per block
int grid_size = 32;             // Block sayısı (optimized kernel için)
```

### Test Verileri

`main.cpp` dosyasında:
```cpp
const int DATA_SIZE = 1000000;  // Test veri boyutu
const int NUM_BINS = 256;       // Histogram bin sayısı
```

## 🧮 Algoritma Detayları

### CPU Single Thread
- Klasik döngü tabanlı histogram hesaplama
- O(n) zaman karmaşıklığı
- Tek çekirdek kullanımı

### CPU Multi-Thread
- OpenMP tabanlı paralel hesaplama
- Thread başına yerel histogram
- Sonuçların birleştirilmesi

### CUDA Basic
- Her thread bir veri elemanını işler
- Global memory atomik operasyonlar
- Basit paralel pattern

### CUDA Optimized
- Shared memory kullanımı
- Grid-stride loop pattern
- Block düzeyinde atomik operasyonlar
- Daha iyi memory coalescing

## 📈 Performans İpuçları

1. **Veri Boyutu**: Büyük veri setlerinde CUDA daha avantajlı
2. **Bin Sayısı**: Çok fazla bin shared memory limitini aşabilir
3. **Block Size**: 256 genellikle optimal (GPU'ya göre değişebilir)
4. **Grid Size**: Optimized kernel için 16-32 block optimal

## 🔍 Hata Ayıklama

### Yaygın Hatalar

1. **CUDA Out of Memory**: Veri boyutunu küçültün
2. **Shared Memory Limit**: Bin sayısını azaltın
3. **Compute Capability**: GPU'nuz CUDA 8.9 desteklemiyorsa CMakeLists.txt'de değiştirin

### Debug Modunda Çalıştırma

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
cuda-gdb ./cuda_histogram
```

## 📚 Öğrenme Kaynakları

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Histogram Algorithms](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında dağıtılmaktadır. Detaylar için `LICENSE` dosyasına bakın.

## ✨ Gelecek Geliştirmeler

- [ ] Multi-GPU desteği
- [ ] Daha büyük veri tipleri (float, double)
- [ ] 2D/3D histogram desteği
- [ ] OpenCV entegrasyonu
- [ ] Benchmark suite
- [ ] Python binding'ler

## 📞 İletişim

Sorularınız için GitHub Issues kullanabilirsiniz.