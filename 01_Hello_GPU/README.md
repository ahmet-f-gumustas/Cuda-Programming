# 01_HelloGPU - CUDA ile Basit Bir GPU Programı

Bu proje, CUDA kullanarak bir GPU programı çalıştırmak için basit bir örnektir. Bu proje, CMake ile derlenir ve CUDA programları oluşturmak için gerekli adımları içerir.

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki araçların sisteminizde kurulu olduğundan emin olun:

- **CUDA Toolkit**: CUDA geliştirme ortamı.
- **CMake**: Derleme sistemini yönetmek için.
- **NVIDIA GPU**: CUDA programlarının çalıştırılması için gerekli.

### Gereken yazılım versiyonları:

- **CUDA Toolkit**: 11.0 veya üstü
- **CMake**: 3.10 veya üstü

## Kurulum

Bu adımlar, projenin derlenmesi ve çalıştırılması için gereken adımları içerir.

1. **Proje dizinini klonlayın veya oluşturun**:
   Eğer bu projeyi bir git reposundan klonladıysanız, ilk adıma geçebilirsiniz. Aksi takdirde proje dosyasını kendiniz oluşturun:

   ```bash
   mkdir 01_HelloGPU
   cd 01_HelloGPU
   cmake ..
   make
   ```