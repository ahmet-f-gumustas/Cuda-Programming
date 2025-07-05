# CUDA Memory Management and Optimization

Bu proje, CUDA'da farklı bellek türlerinin kullanımını ve performans optimizasyonlarını göstermektedir.

## Özellikler

- **Global Memory**: Coalesced vs Non-coalesced memory access karşılaştırması
- **Shared Memory**: Matrix transpose ve reduction örnekleri
- **Constant Memory**: Convolution işlemleri için constant memory kullanımı
- **Unified Memory**: Prefetch ve memory advice optimizasyonları

## Gereksinimler

- CUDA 12.0 veya üzeri
- CMake 3.18 veya üzeri
- C++17 destekleyen bir derleyici

## Derleme ve Çalıştırma

```bash
# Projeyi derle
chmod +x scripts/run_benchmarks.sh
./scripts/run_benchmarks.sh

# Manuel derleme
mkdir build
cd build
cmake ..
make
./memory_benchmark
Bellek Türleri
1. Global Memory

En büyük ama en yavaş bellek türü
Coalesced access ile performans önemli ölçüde artırılabilir
L1/L2 cache desteği

2. Shared Memory

Block içindeki thread'ler arasında paylaşılan hızlı bellek
Bank conflict'lerden kaçınmak önemli
Tile-based algoritmalar için ideal

3. Constant Memory

Read-only, broadcast özelliği olan bellek
Tüm thread'lerin aynı adresi okuması durumunda çok verimli
64KB boyut limiti

4. Unified Memory

CPU ve GPU arasında otomatik veri transferi
Prefetch ve memory advice ile optimize edilebilir
Kod karmaşıklığını azaltır

Performans İpuçları

Memory Coalescing: Sequential memory access pattern kullanın
Bank Conflicts: Shared memory'de stride access'ten kaçının
Occupancy: Register ve shared memory kullanımını dengeleyin
Memory Throughput: Bandwidth kullanımını maksimize edin

Profiling
Performans analizi için:

nvprof (legacy)
nsys (Nsight Systems)
ncu (Nsight Compute)

Örnek Çıktı
