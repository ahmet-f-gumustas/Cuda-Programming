#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <thread>
#include <memory>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "gpu_pipeline.h"
#include "cpu_pipeline.h"
#include "performance_monitor.h"
#include "cuda_edge_detector.h"

class PipelineController {
private:
    std::unique_ptr<GPUPipeline> gpu_pipeline;
    std::unique_ptr<CPUPipeline> cpu_pipeline;
    std::unique_ptr<PerformanceMonitor> perf_monitor;
    std::unique_ptr<CudaEdgeDetector> edge_detector;
    
    std::string input_file;
    std::string output_gpu_file;
    std::string output_cpu_file;
    std::string stats_csv_file;
    
public:
    PipelineController(const std::string& input, const std::string& output_gpu, 
                      const std::string& output_cpu, const std::string& stats_csv)
        : input_file(input), output_gpu_file(output_gpu), 
          output_cpu_file(output_cpu), stats_csv_file(stats_csv) {
        
        gpu_pipeline = std::make_unique<GPUPipeline>();
        cpu_pipeline = std::make_unique<CPUPipeline>();
        perf_monitor = std::make_unique<PerformanceMonitor>(stats_csv);
        edge_detector = std::make_unique<CudaEdgeDetector>();
    }
    
    bool initialize() {
        std::cout << "Initializing pipelines..." << std::endl;
        
        if (!gpu_pipeline->initialize()) {
            std::cerr << "Failed to initialize GPU pipeline" << std::endl;
            return false;
        }
        
        if (!cpu_pipeline->initialize()) {
            std::cerr << "Failed to initialize CPU pipeline" << std::endl;
            return false;
        }
        
        if (!edge_detector->initialize()) {
            std::cerr << "Failed to initialize CUDA edge detector" << std::endl;
            return false;
        }
        
        perf_monitor->start_monitoring();
        std::cout << "All components initialized successfully" << std::endl;
        return true;
    }
    
    void run_gpu_test() {
        std::cout << "\n=== Running GPU Pipeline Test ===" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        perf_monitor->mark_event("GPU_START");
        
        // GPU pipeline ile decode + edge detection + encode
        gpu_pipeline->set_input_file(input_file);
        gpu_pipeline->set_output_file(output_gpu_file);
        gpu_pipeline->set_edge_detector(edge_detector.get());
        
        bool success = gpu_pipeline->process();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        perf_monitor->mark_event("GPU_END");
        
        std::cout << "GPU Pipeline completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Success: " << (success ? "YES" : "NO") << std::endl;
    }
    
    void run_cpu_test() {
        std::cout << "\n=== Running CPU Pipeline Test ===" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        perf_monitor->mark_event("CPU_START");
        
        // CPU pipeline ile decode + edge detection + encode
        cpu_pipeline->set_input_file(input_file);
        cpu_pipeline->set_output_file(output_cpu_file);
        
        bool success = cpu_pipeline->process();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        perf_monitor->mark_event("CPU_END");
        
        std::cout << "CPU Pipeline completed in: " << duration.count() << " ms" << std::endl;
        std::cout << "Success: " << (success ? "YES" : "NO") << std::endl;
    }
    
    void generate_report() {
        std::cout << "\n=== Generating Performance Report ===" << std::endl;
        
        perf_monitor->stop_monitoring();
        perf_monitor->save_csv();
        
        // Python script'i çalıştır grafikler için
        std::string python_cmd = "python3 generate_graphs.py " + stats_csv_file;
        int result = system(python_cmd.c_str());
        
        if (result == 0) {
            std::cout << "Performance graphs generated successfully!" << std::endl;
            std::cout << "Check gpu_vs_cpu_power.png for results" << std::endl;
        } else {
            std::cerr << "Failed to generate graphs. Make sure Python dependencies are installed." << std::endl;
        }
        
        // Özet rapor yazdır
        print_summary_report();
    }
    
private:
    void print_summary_report() {
        std::cout << "\n=== PERFORMANCE SUMMARY REPORT ===" << std::endl;
        std::cout << "Input file: " << input_file << std::endl;
        std::cout << "GPU output: " << output_gpu_file << std::endl;
        std::cout << "CPU output: " << output_cpu_file << std::endl;
        std::cout << "Stats CSV: " << stats_csv_file << std::endl;
        std::cout << "\nFor detailed analysis, check the generated CSV and PNG files." << std::endl;
        std::cout << "=================================" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "CUDA/GPU Accelerated Encode-Decode Pipeline" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // GStreamer'ı başlat
    gst_init(&argc, &argv);
    
    // CUDA device kontrolü
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Varsayılan dosya isimleri
    std::string input_file = "input_1080p60.h264";
    std::string output_gpu_file = "output_gpu_processed.h264";
    std::string output_cpu_file = "output_cpu_processed.h264";
    std::string stats_csv_file = "performance_stats.csv";
    
    // Komut satırı argümanları
    if (argc > 1) input_file = argv[1];
    if (argc > 2) output_gpu_file = argv[2];
    if (argc > 3) output_cpu_file = argv[3];
    if (argc > 4) stats_csv_file = argv[4];
    
    // Test dosyası varlığını kontrol et
    std::ifstream test_file(input_file);
    if (!test_file.good()) {
        std::cout << "Input file not found: " << input_file << std::endl;
        std::cout << "Creating a test video file..." << std::endl;
        
        // Basit test videosu oluştur
        std::string create_cmd = "gst-launch-1.0 videotestsrc num-buffers=1800 ! "
                                "video/x-raw,width=1920,height=1080,framerate=60/1 ! "
                                "x264enc ! h264parse ! filesink location=" + input_file;
        
        if (system(create_cmd.c_str()) != 0) {
            std::cerr << "Failed to create test video. Please provide a valid H264 file." << std::endl;
            return -1;
        }
        
        std::cout << "Test video created: " << input_file << std::endl;
    }
    
    // Pipeline controller'ı başlat
    PipelineController controller(input_file, output_gpu_file, output_cpu_file, stats_csv_file);
    
    if (!controller.initialize()) {
        std::cerr << "Failed to initialize pipeline controller" << std::endl;
        return -1;
    }
    
    try {
        // GPU test'i çalıştır
        controller.run_gpu_test();
        
        // Kısa bekleme
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // CPU test'i çalıştır
        controller.run_cpu_test();
        
        // Rapor oluştur
        controller.generate_report();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during pipeline execution: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\nPipeline tests completed successfully!" << std::endl;
    return 0;
}