#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <thread>
#include <chrono>

// Point3D yapısı
struct Point3D {
    float x, y, z;
    float intensity;
};

// CUDA kernel: Pointcloud verilerini işle
__global__ void processPointCloud(Point3D* points, int numPoints, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numPoints) {
        // Mesafe hesaplama (orijinden)
        float distance = sqrtf(points[idx].x * points[idx].x + 
                              points[idx].y * points[idx].y + 
                              points[idx].z * points[idx].z);
        
        // Threshold'a göre intensity güncelle
        if (distance > threshold) {
            points[idx].intensity = 1.0f;
        } else {
            points[idx].intensity = 0.5f;
        }
        
        // Basit filtreleme - çok uzak noktaları sıfırla
        if (distance > 100.0f) {
            points[idx].x = 0.0f;
            points[idx].y = 0.0f;
            points[idx].z = 0.0f;
            points[idx].intensity = 0.0f;
        }
    }
}

class PointCloudProcessor {
private:
    GstElement *pipeline;
    GstElement *appsrc;
    GstElement *appsink;
    GstBus *bus;
    
    std::vector<Point3D> hostPoints;
    Point3D *devicePoints;
    int numPoints;
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
    
public:
    PointCloudProcessor(int pointCount = 1000) : 
        numPoints(pointCount), 
        rng(42), 
        dist(-50.0f, 50.0f) {
        
        // Host vektörü başlat
        hostPoints.resize(numPoints);
        generateRandomPointCloud();
        
        // CUDA bellek ayır
        cudaMalloc(&devicePoints, numPoints * sizeof(Point3D));
        
        // GStreamer pipeline oluştur
        setupGStreamerPipeline();
    }
    
    ~PointCloudProcessor() {
        if (devicePoints) {
            cudaFree(devicePoints);
        }
        if (pipeline) {
            gst_object_unref(pipeline);
        }
    }
    
    void generateRandomPointCloud() {
        for (int i = 0; i < numPoints; i++) {
            hostPoints[i].x = dist(rng);
            hostPoints[i].y = dist(rng);
            hostPoints[i].z = dist(rng);
            hostPoints[i].intensity = 0.0f;
        }
    }
    
    void setupGStreamerPipeline() {
        // Pipeline oluştur
        pipeline = gst_pipeline_new("pointcloud-pipeline");
        
        // Elementleri oluştur
        appsrc = gst_element_factory_make("appsrc", "source");
        GstElement *queue1 = gst_element_factory_make("queue", "queue1");
        GstElement *capsfilter = gst_element_factory_make("capsfilter", "capsfilter");
        GstElement *queue2 = gst_element_factory_make("queue", "queue2");
        appsink = gst_element_factory_make("appsink", "sink");
        
        if (!pipeline || !appsrc || !queue1 || !capsfilter || !queue2 || !appsink) {
            g_error("GStreamer elementleri oluşturulamadı!");
        }
        
        // Caps ayarla (custom format)
        GstCaps *caps = gst_caps_new_simple("application/x-pointcloud",
                                           "format", G_TYPE_STRING, "xyz-intensity",
                                           "points", G_TYPE_INT, numPoints,
                                           NULL);
        g_object_set(capsfilter, "caps", caps, NULL);
        gst_caps_unref(caps);
        
        // AppSrc ayarları
        g_object_set(appsrc,
                    "caps", caps,
                    "format", GST_FORMAT_TIME,
                    "is-live", TRUE,
                    NULL);
        
        // AppSink ayarları
        g_object_set(appsink,
                    "emit-signals", TRUE,
                    "max-buffers", 1,
                    "drop", TRUE,
                    NULL);
        
        // Pipeline'a ekle
        gst_bin_add_many(GST_BIN(pipeline), appsrc, queue1, capsfilter, queue2, appsink, NULL);
        
        // Elementleri bağla
        if (!gst_element_link_many(appsrc, queue1, capsfilter, queue2, appsink, NULL)) {
            g_error("GStreamer elementleri bağlanamadı!");
        }
        
        // Bus mesajları için
        bus = gst_element_get_bus(pipeline);
    }
    
    void processWithCuda(float threshold = 25.0f) {
        // Veriyi GPU'ya kopyala
        cudaMemcpy(devicePoints, hostPoints.data(), 
                  numPoints * sizeof(Point3D), cudaMemcpyHostToDevice);
        
        // CUDA kernel parametreleri
        int blockSize = 256;
        int gridSize = (numPoints + blockSize - 1) / blockSize;
        
        // Kernel çalıştır
        processPointCloud<<<gridSize, blockSize>>>(devicePoints, numPoints, threshold);
        
        // GPU'dan veriyi geri al
        cudaMemcpy(hostPoints.data(), devicePoints, 
                  numPoints * sizeof(Point3D), cudaMemcpyDeviceToHost);
        
        // CUDA hatalarını kontrol et
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        }
    }
    
    void pushDataToGStreamer() {
        // Buffer oluştur
        GstBuffer *buffer = gst_buffer_new_allocate(NULL, 
                                                   numPoints * sizeof(Point3D), NULL);
        
        // Veriyi buffer'a kopyala
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, hostPoints.data(), numPoints * sizeof(Point3D));
        gst_buffer_unmap(buffer, &map);
        
        // Timestamp ekle
        static guint64 timestamp = 0;
        GST_BUFFER_PTS(buffer) = timestamp;
        GST_BUFFER_DURATION(buffer) = GST_SECOND / 30; // 30 FPS
        timestamp += GST_BUFFER_DURATION(buffer);
        
        // Buffer'ı push et
        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        if (ret != GST_FLOW_OK) {
            std::cerr << "GStreamer buffer push hatası!" << std::endl;
        }
    }
    
    void printStats() {
        int activePoints = 0;
        float avgIntensity = 0.0f;
        float maxDistance = 0.0f;
        
        for (const auto& point : hostPoints) {
            if (point.intensity > 0.0f) {
                activePoints++;
                avgIntensity += point.intensity;
                
                float dist = sqrtf(point.x * point.x + point.y * point.y + point.z * point.z);
                if (dist > maxDistance) {
                    maxDistance = dist;
                }
            }
        }
        
        if (activePoints > 0) {
            avgIntensity /= activePoints;
        }
        
        std::cout << "PointCloud Stats:" << std::endl;
        std::cout << "  Toplam nokta: " << numPoints << std::endl;
        std::cout << "  Aktif nokta: " << activePoints << std::endl;
        std::cout << "  Ortalama intensity: " << avgIntensity << std::endl;
        std::cout << "  Max mesafe: " << maxDistance << std::endl;
        std::cout << "------------------------" << std::endl;
    }
    
    void run() {
        // Pipeline'ı başlat
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
        
        std::cout << "PointCloud işleme başladı..." << std::endl;
        
        for (int frame = 0; frame < 100; frame++) {
            // Yeni rastgele pointcloud oluştur
            generateRandomPointCloud();
            
            // CUDA ile işle
            float threshold = 20.0f + (frame % 20); // Dinamik threshold
            processWithCuda(threshold);
            
            // GStreamer'a gönder
            pushDataToGStreamer();
            
            // İstatistikleri yazdır
            if (frame % 10 == 0) {
                printStats();
            }
            
            // Bus mesajlarını kontrol et
            GstMessage *msg = gst_bus_try_pop(bus);
            if (msg) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                    GError *error;
                    gchar *debug_info;
                    gst_message_parse_error(msg, &error, &debug_info);
                    std::cerr << "GStreamer Error: " << error->message << std::endl;
                    g_error_free(error);
                    g_free(debug_info);
                }
                gst_message_unref(msg);
            }
            
            // FPS kontrol et
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
        }
        
        // Pipeline'ı durdur
        gst_element_set_state(pipeline, GST_STATE_NULL);
        std::cout << "İşlem tamamlandı!" << std::endl;
    }
};

// Signal handler callback (opsiyonel)
static GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (sample) {
        GstBuffer *buffer = gst_sample_get_buffer(sample);
        GstMapInfo map;
        
        if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
            Point3D *points = (Point3D*)map.data;
            int numPoints = map.size / sizeof(Point3D);
            
            std::cout << "AppSink'ten " << numPoints << " nokta alındı" << std::endl;
            
            gst_buffer_unmap(buffer, &map);
        }
        
        gst_sample_unref(sample);
    }
    
    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    // GStreamer'ı başlat
    gst_init(&argc, &argv);
    
    // CUDA cihazını kontrol et
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "CUDA destekli GPU bulunamadı!" << std::endl;
        return -1;
    }
    
    std::cout << "CUDA GPU sayısı: " << deviceCount << std::endl;
    
    // Processor oluştur ve çalıştır
    try {
        PointCloudProcessor processor(2000); // 2000 nokta
        processor.run();
    } catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << std::endl;
        return -1;
    }
    
    // GStreamer'ı temizle
    gst_deinit();
    
    return 0;
}