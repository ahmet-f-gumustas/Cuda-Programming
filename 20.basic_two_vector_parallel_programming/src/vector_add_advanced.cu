// src/vector_add_advanced.cu - Advanced Version with Performance Analysis

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA kernel fonksiyonu - iki vektörü paralel olarak toplar
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Thread indexing - her thread hangi elemanı işleyeceğini hesaplar
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Sınır kontrolü - vektör boyutunu aşmamak için
    if (index < n) {
        c[index] = a[index] + b[index];
    }
    
    // Thread senkronizasyonu (block içindeki tüm threadler burada bekler)
    __syncthreads();
}

// CPU'da vektör toplama (doğrulama için)
void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Sonuçları karşılaştırma fonksiyonu
bool compareResults(float *cpu_result, float *gpu_result, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Hata: Index %d'de farklılık - CPU: %f, GPU: %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Vektör boyutu (en az 1024 eleman)
    const int N = 2048;
    const int size = N * sizeof(float);
    
    printf("CUDA Vektör Toplama Programı\n");
    printf("Vektör boyutu: %d eleman\n", N);
    printf("Toplam bellek kullanımı: %.2f MB\n", (3 * size) / (1024.0 * 1024.0));
    
    // Host (CPU) bellek alanları
    float *h_a, *h_b, *h_c, *h_c_cpu;
    
    // Host bellek ayırma
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    
    if (!h_a || !h_b || !h_c || !h_c_cpu) {
        printf("Host bellek ayırma hatası!\n");
        return -1;
    }
    
    // Vektörleri rastgele değerlerle doldur
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX * 100.0f;  // 0-100 arası rastgele değer
        h_b[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    // Device (GPU) bellek alanları
    float *d_a, *d_b, *d_c;
    
    // GPU bellek ayırma
    cudaError_t err;
    err = cudaMalloc((void**)&d_a, size);
    if (err != cudaSuccess) {
        printf("GPU bellek ayırma hatası (d_a): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&d_b, size);
    if (err != cudaSuccess) {
        printf("GPU bellek ayırma hatası (d_b): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void**)&d_c, size);
    if (err != cudaSuccess) {
        printf("GPU bellek ayırma hatası (d_c): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Veriyi host'tan device'a kopyala
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // CUDA kernel parametreleri
    int threadsPerBlock = 256;  // Her block'ta 256 thread
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Gerekli block sayısı
    
    printf("\nKernel Parametreleri:\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    printf("Toplam thread sayısı: %d\n", threadsPerBlock * blocksPerGrid);
    
    // CUDA event'leri ile zaman ölçümü
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // GPU hesaplama başlat
    printf("\nGPU hesaplama başlatılıyor...\n");
    cudaEventRecord(start, 0);
    
    // CUDA kernel çağrısı
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Kernel tamamlanmasını bekle
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // GPU süresini hesapla
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    // Kernel hata kontrolü
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel çalışma hatası: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Sonucu device'dan host'a kopyala
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // CPU ile doğrulama hesaplaması
    printf("CPU doğrulama hesaplaması...\n");
    clock_t cpu_start = clock();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    clock_t cpu_end = clock();
    
    double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    // Sonuçları karşılaştır
    printf("\nSonuç Doğrulama:\n");
    if (compareResults(h_c_cpu, h_c, N)) {
        printf("✓ GPU ve CPU sonuçları eşleşiyor!\n");
    } else {
        printf("✗ GPU ve CPU sonuçları eşleşmiyor!\n");
    }
    
    // İlk birkaç elemanı göster
    printf("\nÖrnek Sonuçlar (ilk 10 eleman):\n");
    printf("Index\tA[i]\t\tB[i]\t\tC[i] (GPU)\tC[i] (CPU)\n");
    printf("-----\t--------\t--------\t----------\t----------\n");
    for (int i = 0; i < 10; i++) {
        printf("%d\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n", 
               i, h_a[i], h_b[i], h_c[i], h_c_cpu[i]);
    }
    
    // Performans sonuçları
    printf("\nPerformans Sonuçları:\n");
    printf("GPU Süresi: %.3f ms\n", gpu_time);
    printf("CPU Süresi: %.3f ms\n", cpu_time);
    printf("Hızlanma: %.2fx\n", cpu_time / gpu_time);
    printf("Throughput: %.2f GFLOPS\n", (N * 1e-9) / (gpu_time * 1e-3));
    
    // Bellek temizleme
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nProgram başarıyla tamamlandı!\n");
    
    return 0;
}