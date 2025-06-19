# GStreamer CUDA PointCloud Processor

Bu proje, GStreamer ve CUDA kullanarak 3D pointcloud verilerini işleyen basit bir C++ uygulamasıdır. Vektör işlemleri ile pointcloud verilerini simüle eder ve GPU üzerinde paralel olarak işler.

## Özellikler

- **CUDA Vektör İşlemleri**: 3D pointcloud verilerini GPU üzerinde paralel işleme
- **GStreamer Pipeline**: Veri akışı yönetimi için GStreamer kullanımı
- **Rastgele PointCloud Üretimi**: Simülasyon için rastgele 3D nokta bulutları
- **Gerçek Zamanlı İşleme**: ~30 FPS ile sürekli veri işleme
- **Mesafe Filtreleme**: Threshold tabanlı nokta filtreleme
- **İstatistik Raporlama**: İşlenen veri hakkında anlık istatistikler

## Sistem Gereksinimleri

- Ubuntu 22.04 (veya benzer Linux dağıtımı)
- NVIDIA GPU (CUDA desteği ile)
- CUDA Toolkit 12.4+
- GStreamer 1.0+
- CMake 3.18+
- GCC/G++ 9.0+

## Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
sudo apt-get update
sudo apt-get install \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    cmake \
    build-essential \
    pkg-config
```

### 2. Projeyi Derleyin

```bash
# Build script'ini çalıştırılabilir yapın
chmod +x build.sh

# Projeyi derleyin
./build.sh
```

Veya manuel olarak:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Kullanım

```bash
# Build dizininden çalıştırın
cd build
./pointcloud_processor

# Veya doğrudan
./build/pointcloud_processor
```

## Proje Yapısı

```
├── main.cu              # Ana kaynak kod
├── CMakeLists.txt       # CMake yapılandırması
├── build.sh             # Build script'i
└── README.md            # Bu dosya
```

## Kod Açıklaması

### CUDA Kernel
```cpp
__global__ void processPointCloud(Point3D* points, int numPoints, float threshold)
```
- Her GPU thread'i bir 3D noktayı işler
- Orijinden mesafe hesaplar
- Threshold'a göre intensity değerini günceller
- Çok uzak noktaları filtreler

### GStreamer Pipeline
```
appsrc -> queue -> capsfilter -> queue -> appsink
```
- `appsrc`: Veri girişi
- `capsfilter`: Özel pointcloud formatı
- `appsink`: İşlenmiş veri çıkışı

### Vektör İşlemleri
- `std::vector<Point3D>` ile host bellek yönetimi
- CUDA memory transfer'i (Host ↔ Device)
- Paralel vektör işleme

## Performans

RTX 4070 üzerinde:
- 2000 nokta/frame
- ~30 FPS
- GPU bellek kullanımı: ~1MB
- İşleme süresi: <1ms/frame

## Özelleştirme

### Nokta Sayısını Değiştirme
```cpp
PointCloudProcessor processor(5000); // 5000 nokta
```

### Threshold Ayarlama
```cpp
processWithCuda(30.0f); // 30.0 birim threshold
```

### FPS Ayarlama
```cpp
std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
```

## Hata Ayıklama

### CUDA Hataları
```bash
# CUDA cihazlarını kontrol edin
nvidia-smi

# CUDA compiler versiyonunu kontrol edin
nvcc --version
```

### GStreamer Hataları
```bash
# GStreamer plugin'lerini kontrol edin
gst-inspect-1.0 appsrc
gst-inspect-1.0 appsink

# Debug modu için
export GST_DEBUG=3
./pointcloud_processor
```

## Geliştirme

Bu proje, aşağıdaki konularda örnek teşkil eder:
- CUDA ve C++ entegrasyonu
- GStreamer pipeline tasarımı
- Vektör tabanlı veri işleme
- GPU bellek yönetimi
- Gerçek zamanlı veri akışı

## Lisans

Bu proje eğitim amaçlıdır ve özgürce kullanılabilir.

## Sorun Giderme

### Derleme Hataları
- CUDA path'inin doğru olduğundan emin olun
- GStreamer development paketlerini kontrol edin
- CMake versiyonunu kontrol edin

### Çalışma Zamanı Hataları
- GPU belleğinin yeterli olduğundan emin olun
- GStreamer pipeline'ının doğru kurulduğunu kontrol edin
- CUDA driver versiyonunu kontrol edin