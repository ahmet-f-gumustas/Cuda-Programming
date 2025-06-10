#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>

struct PerformanceData {
    std::chrono::time_point<std::chrono::high_resolution_clock> timestamp;
    double cpu_usage_percent;
    double gpu_usage_percent;
    double gpu_memory_usage_mb;
    double total_memory_usage_mb;
    double power_consumption_watts;
    double temperature_celsius;
    double fps;
    std::string event_name;
    
    PerformanceData() 
        : cpu_usage_percent(0), gpu_usage_percent(0), gpu_memory_usage_mb(0),
          total_memory_usage_mb(0), power_consumption_watts(0), 
          temperature_celsius(0), fps(0) {}
};

class PerformanceMonitor {
private:
    std::string csv_filename;
    std::vector<PerformanceData> performance_log;
    std::atomic<bool> monitoring_active;
    std::thread monitoring_thread;
    std::thread tegrastats_thread;
    std::mutex data_mutex;
    
    // Tegrastats parsing
    std::string tegrastats_output_file;
    bool parse_tegrastats_line(const std::string& line, PerformanceData& data);
    
    // System monitoring functions
    double get_cpu_usage();
    double get_gpu_usage();
    double get_gpu_memory_usage();
    double get_total_memory_usage();
    double get_power_consumption();
    double get_temperature();
    
    // Monitoring loop
    void monitoring_loop();
    void tegrastats_monitoring_loop();
    
public:
    PerformanceMonitor(const std::string& csv_file);
    ~PerformanceMonitor();
    
    void start_monitoring();
    void stop_monitoring();
    
    void mark_event(const std::string& event_name);
    void save_csv();
    
    // Get current stats
    PerformanceData get_current_stats();
    
    // Static utility functions
    static bool is_jetson_platform();
    static bool start_tegrastats(const std::string& output_file);
    static void stop_tegrastats();
};

#endif // PERFORMANCE_MONITOR_H