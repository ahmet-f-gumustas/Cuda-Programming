#include "performance_monitor.h"
#include <iostream>
#include <sstream>
#include <regex>
#include <fstream>
#include <iomanip>
#include <sys/sysinfo.h>
#include <unistd.h>

PerformanceMonitor::PerformanceMonitor(const std::string& csv_file)
    : csv_filename(csv_file), monitoring_active(false),
      tegrastats_output_file("tegrastats_output.log") {
}

PerformanceMonitor::~PerformanceMonitor() {
    stop_monitoring();
}

void PerformanceMonitor::start_monitoring() {
    if (monitoring_active.load()) {
        std::cout << "Performance monitoring already active" << std::endl;
        return;
    }
    
    std::cout << "Starting performance monitoring..." << std::endl;
    monitoring_active.store(true);
    
    // Tegrastats'ı başlat (Jetson platformunda)
    if (is_jetson_platform()) {
        std::cout << "Jetson platform detected, starting tegrastats" << std::endl;
        start_tegrastats(tegrastats_output_file);
        
        // Tegrastats monitoring thread
        tegrastats_thread = std::thread(&PerformanceMonitor::tegrastats_monitoring_loop, this);
    }
    
    // Genel monitoring thread
    monitoring_thread = std::thread(&PerformanceMonitor::monitoring_loop, this);
    
    std::cout << "Performance monitoring started" << std::endl;
}

void PerformanceMonitor::stop_monitoring() {
    if (!monitoring_active.load()) {
        return;
    }
    
    std::cout << "Stopping performance monitoring..." << std::endl;
    monitoring_active.store(false);
    
    // Threads'leri bekle
    if (monitoring_thread.joinable()) {
        monitoring_thread.join();
    }
    
    if (tegrastats_thread.joinable()) {
        tegrastats_thread.join();
    }
    
    // Tegrastats'ı durdur
    if (is_jetson_platform()) {
        stop_tegrastats();
    }
    
    std::cout << "Performance monitoring stopped" << std::endl;
}

void PerformanceMonitor::monitoring_loop() {
    while (monitoring_active.load()) {
        PerformanceData data;
        data.timestamp = std::chrono::high_resolution_clock::now();
        
        // System stats topla
        data.cpu_usage_percent = get_cpu_usage();
        data.gpu_usage_percent = get_gpu_usage();
        data.gpu_memory_usage_mb = get_gpu_memory_usage();
        data.total_memory_usage_mb = get_total_memory_usage();
        data.power_consumption_watts = get_power_consumption();
        data.temperature_celsius = get_temperature();
        
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            performance_log.push_back(data);
        }
        
        // 100ms aralıkla sample al
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void PerformanceMonitor::tegrastats_monitoring_loop() {
    std::ifstream tegrastats_file;
    std::string line;
    
    while (monitoring_active.load()) {
        tegrastats_file.open(tegrastats_output_file, std::ios::in);
        
        if (tegrastats_file.is_open()) {
            // Dosyanın sonuna git
            tegrastats_file.seekg(0, std::ios::end);
            
            while (monitoring_active.load() && std::getline(tegrastats_file, line)) {
                PerformanceData tegra_data;
                tegra_data.timestamp = std::chrono::high_resolution_clock::now();
                
                if (parse_tegrastats_line(line, tegra_data)) {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    performance_log.push_back(tegra_data);
                }
            }
            
            tegrastats_file.close();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

bool PerformanceMonitor::parse_tegrastats_line(const std::string& line, PerformanceData& data) {
    // Tegrastats output örneği:
    // 05-30-2024 15:30:45 RAM 3963/7846MB (lfb 1018x4MB) SWAP 0/3923MB (cached 0MB) CPU [25%@1428,19%@1428,21%@1428,18%@1428] EMC_FREQ 0% GR3D_FREQ 76% VIC_FREQ 0% APE 150 MTS fg 0% bg 0% PLL@45C MCPU@45.5C PMIC@100C Tboard@44C GPU@44C BCPU@45C thermal@44.8C VDD_IN 8042/8042 VDD_CPU_GPU_CV 2534/2534 VDD_SOC 1518/1518
    
    try {
        std::regex ram_regex(R"(RAM (\d+)/(\d+)MB)");
        std::regex cpu_regex(R"(CPU \[([^\]]+)\])");
        std::regex gpu_regex(R"(GR3D_FREQ (\d+)%)");
        std::regex temp_regex(R"(GPU@([\d.]+)C)");
        std::regex power_regex(R"(VDD_IN (\d+)/(\d+))");
        
        std::smatch match;
        
        // RAM usage
        if (std::regex_search(line, match, ram_regex)) {
            data.total_memory_usage_mb = std::stod(match[1].str());
        }
        
        // GPU usage
        if (std::regex_search(line, match, gpu_regex)) {
            data.gpu_usage_percent = std::stod(match[1].str());
        }
        
        // Temperature
        if (std::regex_search(line, match, temp_regex)) {
            data.temperature_celsius = std::stod(match[1].str());
        }
        
        // Power consumption
        if (std::regex_search(line, match, power_regex)) {
            data.power_consumption_watts = std::stod(match[1].str()) / 1000.0; // mW to W
        }
        
        // CPU usage (ortalama hesapla)
        if (std::regex_search(line, match, cpu_regex)) {
            std::string cpu_str = match[1].str();
            std::regex cpu_core_regex(R"((\d+)%)");
            std::sregex_iterator iter(cpu_str.begin(), cpu_str.end(), cpu_core_regex);
            std::sregex_iterator end;
            
            double total_cpu = 0.0;
            int core_count = 0;
            
            for (; iter != end; ++iter) {
                total_cpu += std::stod((*iter)[1].str());
                core_count++;
            }
            
            if (core_count > 0) {
                data.cpu_usage_percent = total_cpu / core_count;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing tegrastats line: " << e.what() << std::endl;
        return false;
    }
}

double PerformanceMonitor::get_cpu_usage() {
    static long long prev_idle = 0, prev_total = 0;
    
    std::ifstream stat_file("/proc/stat");
    std::string line;
    
    if (!std::getline(stat_file, line)) {
        return 0.0;
    }
    
    std::istringstream iss(line);
    std::string cpu;
    long long user, nice, system, idle, iowait, irq, softirq, steal;
    
    iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    
    long long current_idle = idle + iowait;
    long long current_total = user + nice + system + idle + iowait + irq + softirq + steal;
    
    long long idle_diff = current_idle - prev_idle;
    long long total_diff = current_total - prev_total;
    
    double usage = 0.0;
    if (total_diff > 0) {
        usage = 100.0 * (1.0 - (double)idle_diff / total_diff);
    }
    
    prev_idle = current_idle;
    prev_total = current_total;
    
    return usage;
}

double PerformanceMonitor::get_gpu_usage() {
    // nvidia-smi kullanarak GPU usage al
    FILE* pipe = popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", "r");
    if (!pipe) {
        return 0.0;
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    try {
        return std::stod(result);
    } catch (...) {
        return 0.0;
    }
}

double PerformanceMonitor::get_gpu_memory_usage() {
    FILE* pipe = popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", "r");
    if (!pipe) {
        return 0.0;
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    try {
        return std::stod(result);
    } catch (...) {
        return 0.0;
    }
}

double PerformanceMonitor::get_total_memory_usage() {
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        long long used_ram = si.totalram - si.freeram;
        return (double)used_ram / (1024 * 1024); // Bytes to MB
    }
    return 0.0;
}

double PerformanceMonitor::get_power_consumption() {
    // nvidia-smi ile power consumption al
    FILE* pipe = popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", "r");
    if (!pipe) {
        return 0.0;
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    try {
        return std::stod(result);
    } catch (...) {
        return 0.0;
    }
}

double PerformanceMonitor::get_temperature() {
    FILE* pipe = popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", "r");
    if (!pipe) {
        return 0.0;
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    
    try {
        return std::stod(result);
    } catch (...) {
        return 0.0;
    }
}

void PerformanceMonitor::mark_event(const std::string& event_name) {
    PerformanceData event_data = get_current_stats();
    event_data.event_name = event_name;
    
    std::lock_guard<std::mutex> lock(data_mutex);
    performance_log.push_back(event_data);
    
    std::cout << "Event marked: " << event_name << std::endl;
}

PerformanceData PerformanceMonitor::get_current_stats() {
    PerformanceData data;
    data.timestamp = std::chrono::high_resolution_clock::now();
    data.cpu_usage_percent = get_cpu_usage();
    data.gpu_usage_percent = get_gpu_usage();
    data.gpu_memory_usage_mb = get_gpu_memory_usage();
    data.total_memory_usage_mb = get_total_memory_usage();
    data.power_consumption_watts = get_power_consumption();
    data.temperature_celsius = get_temperature();
    
    return data;
}

void PerformanceMonitor::save_csv() {
    std::lock_guard<std::mutex> lock(data_mutex);
    
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
        return;
    }
    
    // Header yaz
    csv_file << "timestamp,cpu_usage_percent,gpu_usage_percent,gpu_memory_usage_mb,"
             << "total_memory_usage_mb,power_consumption_watts,temperature_celsius,"
             << "fps,event_name\n";
    
    // Data yazma
    for (const auto& data : performance_log) {
        auto time_t = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now() + 
            (data.timestamp - std::chrono::high_resolution_clock::now()));
        
        csv_file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                 << data.cpu_usage_percent << ","
                 << data.gpu_usage_percent << ","
                 << data.gpu_memory_usage_mb << ","
                 << data.total_memory_usage_mb << ","
                 << data.power_consumption_watts << ","
                 << data.temperature_celsius << ","
                 << data.fps << ","
                 << data.event_name << "\n";
    }
    
    csv_file.close();
    std::cout << "Performance data saved to: " << csv_filename << std::endl;
    std::cout << "Total data points: " << performance_log.size() << std::endl;
}

bool PerformanceMonitor::is_jetson_platform() {
    std::ifstream model_file("/sys/firmware/devicetree/base/model");
    if (model_file.is_open()) {
        std::string model;
        std::getline(model_file, model);
        return model.find("Jetson") != std::string::npos;
    }
    return false;
}

bool PerformanceMonitor::start_tegrastats(const std::string& output_file) {
    std::string cmd = "tegrastats --logfile " + output_file + " &";
    return system(cmd.c_str()) == 0;
}

void PerformanceMonitor::stop_tegrastats() {
    system("pkill tegrastats");
}