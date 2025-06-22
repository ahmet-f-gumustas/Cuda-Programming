#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

int main() {
    // Vektör boyutu (en az 1024 eleman)
    const int N = 1024;
    const int size = N * sizeof(float);
    
    printf("CUDA Vektör Toplama Programı (Basit Versiyon)\n");
    printf("Vektör boyutu: %d eleman\n", N);
    
    // Host (CPU) bellek alanları
    float *h_a, *h_b, *h_c;
    
    // Host bellek ayırma
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    if (!h_a || !h_b || !h_c) {
        printf("Host bellek ayırma hatası!\n");
        return -1;
    }
    
    // Vektörleri başlat
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    
    // Device (GPU) bellek alanları
    float *d_a, *d_b, *d_c;
    
    // GPU bellek ayırma
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Veriyi host'tan device'a kopyala
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // CUDA kernel parametreleri
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    
    // CUDA kernel çağrısı
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Kernel tamamlanmasını bekle
    cudaDeviceSynchronize();
    
    // Sonucu device'dan host'a kopyala
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // İlk birkaç sonucu kontrol et
    printf("\nİlk 10 sonuç:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.0f + %.0f = %.0f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Bellek temizleme
    free(h_a);
    free(h_b);
    free(h_c);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("\nProgram başarıyla tamamlandı!\n");
    
    return 0;
}