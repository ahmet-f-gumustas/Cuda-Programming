# ParallelVision - CUDA ile Görüntü İşleme

ParallelVision, CUDA ve C++ kullanarak görüntü işleme algoritmalarının GPU üzerinde hızlandırılmasını gösteren bir projedir. Bu proje, çeşitli görüntü işleme filtrelerini hem CPU hem de GPU üzerinde çalıştırarak performans karşılaştırması yapmanıza olanak tanır.

## Özellikler

- Gri tonlama dönüşümü
- Gauss bulanıklaştırma
- Sobel kenar algılama
- Histogram eşitleme
- Görüntü keskinleştirme
- Her işlem için CPU ve GPU implementasyonları
- Performans karşılaştırma araçları

## Gereksinimler

- CUDA 12.0 veya üzeri
- NVIDIA GPU (Compute Capability 6.0+)
- OpenCV 4.x
- CMake 3.10 veya üzeri
- C++17 destekli compiler (GCC, Clang)

## Kurulum

### Bağımlılıkların Kurulumu

Ubuntu üzerinde OpenCV kurulumu:

```bash
sudo apt update
sudo apt install libopencv-dev
```

### Projeyi Derleme

```bash
mkdir build
cd build
cmake ..
make -j4
```

### Çalıştırma

```bash
./parallelvision
```

## Kullanım

Program interaktif bir menü sunmaktadır:

1. İlk olarak bir görüntü yükleyin.
2. Çeşitli görüntü işleme algoritmaları arasından seçim yapın.
3. Seçilen algoritma hem CPU hem de GPU üzerinde çalıştırılır ve performans karşılaştırması gösterilir.
4. "Tüm İşlemleri Çalıştır" seçeneği ile tüm algoritmaların performans karşılaştırmasını görebilirsiniz.

## Örnek Kullanım

```
===== ParallelVision - CUDA Görüntü İşleme =====
1. Görüntü Yükle
2. Gri Tonlama (CPU vs GPU)
3. Gauss Bulanıklaştırma (CPU vs GPU)
4. Sobel Kenar Algılama (CPU vs GPU)
5. Histogram Eşitleme (CPU vs GPU)
6. Keskinleştirme (CPU vs GPU)
7. Tüm İşlemleri Çalıştır ve Performans Karşılaştır
0. Çıkış
Seçiminiz: 1
Görüntü yolunu girin: /path/to/image.jpg
Görüntü başarıyla yüklendi.

===== ParallelVision - CUDA Görüntü İşleme =====
...
Seçiminiz: 7

===== Performans Karşılaştırması =====

1. Gri Tonlama:
CPU Süre: 15 ms
GPU Süre: 0.5 ms
Hızlanma: 30x

2. Gauss Bulanıklaştırma:
CPU Süre: 120 ms
GPU Süre: 3 ms
Hızlanma: 40x

...
```

## Test Edilen Donanım

Bu proje, NVIDIA GeForce RTX 4070 (CUDA 12.4) üzerinde test edilmiştir.

## Proje Yapısı

```
.
├── CMakeLists.txt
├── include
│   ├── cuda_kernels.cuh     # CUDA kernel tanımlamaları
│   └── image_processor.h    # Görüntü işleme sınıfı tanımlaması
├── src
│   ├── cuda_kernels.cu      # CUDA kernel implementasyonları
│   ├── image_processor.cpp  # Görüntü işleme sınıfı implementasyonu
│   └── main.cpp             # Ana program
└── README.md
```

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen bir Pull Request gönderin veya bir Issue açın.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.
