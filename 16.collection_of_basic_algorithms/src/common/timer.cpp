// ==================== src/common/timer.cpp ====================
#include "../../include/performance.h"

// CudaTimer implementation
CudaTimer::CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
}

CudaTimer::~CudaTimer() {
    if (start_event) cudaEventDestroy(start_event);
    if (stop_event) cudaEventDestroy(stop_event);
}

CudaTimer::CudaTimer(CudaTimer&& other) noexcept 
    : start_event(other.start_event), stop_event(other.stop_event), started(other.started) {
    other.start_event = nullptr;
    other.stop_event = nullptr;
    other.started = false;
}

CudaTimer& CudaTimer::operator=(CudaTimer&& other) noexcept {
    if (this != &other) {
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        
        start_event = other.start_event;
        stop_event = other.stop_event;
        started = other.started;
        
        other.start_event = nullptr;
        other.stop_event = nullptr;
        other.started = false;
    }
    return *this;
}

void CudaTimer::start() {
    CUDA_CHECK(cudaEventRecord(start_event, 0));
    started = true;
}

void CudaTimer::stop() {
    CUDA_CHECK(cudaEventRecord(stop_event, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    started = false;
}

float CudaTimer::elapsed_ms() {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
    return ms;
}

void CudaTimer::reset() {
    started = false;
}

// CpuTimer implementation
void CpuTimer::start() {
    start_time = std::chrono::high_resolution_clock::now();
    started = true;
}

void CpuTimer::stop() {
    end_time = std::chrono::high_resolution_clock::now();
    started = false;
}

float CpuTimer::elapsed_ms() {
    auto duration = std::chrono::duration<float, std::milli>(end_time - start_time);
    return duration.count();
}

double CpuTimer::elapsed_us() {
    auto duration = std::chrono::duration<double, std::micro>(end_time - start_time);
    return duration.count();
}
