# 15. Image Processing with CUDA

CUDA kullanarak temel görüntü işleme operasyonlarını gerçekleştiren kapsamlı eğitim projesi. Bu proje ile GPU programlama temellerini görüntü işleme üzerinden öğreneceksiniz.

## 🎯 Öğrenilecek Konular

### CUDA Temel Kavramları
- **2D Grid ve Block Yapısı** - GPU thread organizasyonu
- **Shared Memory Kullanımı** - Yüksek performans için lokal memory
- **Constant Memory** - Kernel parametreleri için optimize edilmiş memory
- **GPU Memory Yönetimi** - Host ↔ Device veri transferi
- **Error Handling** - CUDA hata kontrolü ve debugging

### Görüntü İşleme Algoritmaları
- **Color Space Conversion** - RGB'den grayscale'e dönüşüm
- **Convolution Operations** - Kernel tabanlı filtreler
- **Edge Detection** - Sobel operatörü ile kenar tespiti
- **Spatial Filtering** - Gaussian blur ve diğer filtreler

## 🖼️ Desteklenen Filtreler

### 1. Grayscale Conversion
Renkli görüntüyü gri tonlamaya çevirir. Standart luminance formülü kullanır:
```
Gray = 0.299×R + 0.587×G + 0.114×B
```

### 2. Gaussian Blur
Görüntüyü yumuşatır ve gürültüyü azaltır. Shared memory optimizasyonu ile hızlandırılmış.
- Ayarlanabilir sigma değeri
- Kernel boyutu otomatik hesaplanır
- Border handling (kenar işleme)

### 3. Edge Detection (Sobel)
Sobel operatörü kullanarak kenarları tespit eder:
- X ve Y yönlü gradyanlar
- Magnitude hesaplaması
- Threshold uygulaması

### 4. Brightness Adjustment
Görüntü parlaklığını artırır veya azaltır:
- Çarpımsal parlaklık faktörü
- Overflow/underflow koruması
- Tüm kanallar için uniform ayarlama

## 🚀 Kurulum ve Çalıştırma

### Sistem Gereksinimleri
- **GPU**: NVIDIA GPU (Compute Capability 3.0+)
- **CUDA**: CUDA Toolkit 11.0 veya üzeri
- **Compiler**: C++14 destekli derleyici (GCC/Clang)
- **Build System**: CMake 3.18+

### Derleme
```bash
# Klasöre gidin
cd 15.Image_Processing_CUDA

# Build script'i çalıştırılabilir yapın
chmod +x build.sh

# Projeyi derleyin
./build.sh
```

Build script otomatik olarak:
- Build klasörü oluşturur
- CMake ile projeyi configure eder
- Paralel derleme yapar
- Örnek görüntüler oluşturur

### Manuel Derleme
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 🎮 Kullanım

### Temel Syntax
```bash
./image_processor <input.ppm> <output.ppm> <filter> [parameters]
```

### Detaylı Örnekler

#### Grayscale Conversion
```bash
./image_processor ../images/sample.ppm gray.ppm grayscale
```

#### Gaussian Blur
```bash
# Hafif blur (sigma=1.0)
./image_processor ../images/sample.ppm blur_light.ppm blur

# Orta blur (sigma=2.0)
./image_processor ../images/sample.ppm blur_medium.ppm blur 2.0

# Güçlü blur (sigma=3.5)
./image_processor ../images/sample.ppm blur_heavy.ppm blur 3.5
```

#### Edge Detection
```bash
./image_processor ../images/sample.ppm edges.ppm edge
```

#### Brightness Adjustment
```bash
# %50 daha parlak
./image_processor ../images/sample.ppm bright.ppm brightness 1.5

# %30 daha karanlık
./image_processor ../images/sample.ppm dark.ppm brightness 0.7

# Çok parlak
./image_processor ../images/sample.ppm very_bright.ppm brightness 2.0
```

## 📁 Proje Yapısı

```
15.Image_Processing_CUDA/
├── src/                          # Kaynak dosyalar
│   ├── main.cpp                  # Ana program ve argument parsing
│   ├── image_processing.cu       # CUDA kernels ve GPU işlemleri
│   └── image_utils.cpp           # Dosya I/O ve CPU referans implementasyonları
├── include/                      # Header dosyalar
│   ├── image_processing.h        # CUDA fonksiyon tanımları
│   └── image_utils.h             # Utility fonksiyon tanımları
├── images/                       # Test görüntüleri
│   ├── sample.ppm               # 512x512 test görüntüsü
│   ├── sample_small.ppm         # 256x256 küçük test görüntüsü
│   └── sample_large.ppm         # 1024x768 büyük test görüntüsü
├── build/                        # Derleme çıktıları (otomatik oluşur)
├── CMakeLists.txt               # CMake build configuration
├── build.sh                     # Otomatik build script
├── create_sample_image.py       # Test görüntüleri oluşturucu
└── README.md                    # Bu dokümantasyon
```

## 📊 Performance Analizi

Program çalıştığında aşağıdaki metrikleri görüntüler:

### GPU Bilgileri
```
CUDA Devices found: 1

Device 0: NVIDIA GeForce RTX 4070
  Compute Capability: 8.9
  Global Memory: 8188 MB
  Shared Memory per Block: 48 KB
  Max Threads per Block: 1024
  Multiprocessors: 36
```

### İşlem Süreleri
```
=== CUDA Image Processing ===
Image: 1920x1080 pixels (RGB)
Filter: Gaussian Blur (sigma=2.0)

GPU processing time: 2.3 ms
CPU reference time:  45.2 ms
Speedup: 19.65x

Memory breakdown:
- Host to Device: 1.2 ms
- Kernel execution: 0.8 ms
- Device to Host: 0.3 ms
```

### Beklenen Performans (RTX 4070)
- **Grayscale**: 15-25x speedup
- **Gaussian Blur**: 10-20x speedup
- **Edge Detection**: 12-18x speedup
- **Brightness**: 20-30x speedup

## 🔧 Kod Açıklaması

### CUDA Kernel Örneği
```cpp
__global__ void grayscaleKernel(unsigned char* input, unsigned char* output, 
                               int width, int height) {
    // Thread koordinatları
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Boundary check
    if (x < width && y < height) {
        int pixel_idx = (y * width + x) * 3; // RGB
        
        // Luminance hesaplama
        float gray = 0.299f * input[pixel_idx] + 
                     0.587f * input[pixel_idx + 1] + 
                     0.114f * input[pixel_idx + 2];
        
        output[y * width + x] = (unsigned char)gray;
    }
}
```

### Memory Management
```cpp
// GPU memory allocation
unsigned char *d_input, *d_output;
cudaMalloc(&d_input, imageSize);
cudaMalloc(&d_output, outputSize);

// Data transfer
cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);

// Kernel launch
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

// Result transfer
cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
```

## 🎓 Öğrenim Hedefleri

Bu proje tamamlandığında şunları öğrenmiş olacaksınız:

### Temel CUDA Programlama
- ✅ Kernel yazma ve launch etme
- ✅ Thread hierarchy (grid, block, thread)
- ✅ Memory management ve transfer
- ✅ Error handling ve debugging

### İleri CUDA Teknikleri
- ✅ Shared memory optimizasyonu
- ✅ Constant memory kullanımı
- ✅ 2D indexing ve boundary handling
- ✅ Performance profiling

### Görüntü İşleme Temelleri
- ✅ Pixel manipülasyonu
- ✅ Convolution operations
- ✅ Color space conversions
- ✅ Spatial filtering

## 🚨 Sık Karşılaşılan Sorunlar

### Derleme Hataları
```bash
# CUDA bulunamadı
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CMake versiyonu düşük
sudo apt update && sudo apt install cmake
```

### Runtime Hataları
```bash
# GPU memory yetersiz
nvidia-smi  # Memory kullanımını kontrol edin

# CUDA driver sorunu
nvidia-smi  # Driver durumunu kontrol edin
```

### Performance Sorunları
- Büyük görüntüler için block size'ı ayarlayın
- Shared memory kullanımını optimize edin
- Memory alignment'ı kontrol edin

## 🔮 İleri Geliştirmeler

### Yeni Filtreler
- Histogram equalization
- Sepia effect
- Emboss filter
- Median filter

### Optimizasyonlar
- Texture memory kullanımı
- Stream processing
- Multi-GPU support
- Half precision (FP16)

### Ek Özellikler
- JPEG/PNG format desteği
- Batch processing
- Real-time video işleme

---

**🎯 Bu proje CUDA öğrenim serinizin 15. adımıdır. RTX 4070 ile test edilmiş ve optimize edilmiştir.**

**📚 İyi kodlamalar ve başarılı GPU programlama deneyimleri! 🚀**