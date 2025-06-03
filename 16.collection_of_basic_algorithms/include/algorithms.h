#pragma once

#include "common.h"
#include <vector>
#include <string>

// Forward declarations for all algorithm functions

// Prefix Sum algorithms
void test_prefix_sum();
void prefix_sum_custom(const std::vector<int>& input, std::vector<int>& output, const std::string& method);
void prefix_sum_thrust(const std::vector<int>& input, std::vector<int>& output);
void prefix_sum_cub(const std::vector<int>& input, std::vector<int>& output);

// Reduce algorithms
void test_reduce();
int reduce_custom(const std::vector<int>& input, const std::string& method);
int reduce_thrust(const std::vector<int>& input);
int reduce_cub(const std::vector<int>& input);

// Histogram algorithms
void test_histogram();
void histogram_custom(const std::vector<int>& input, std::vector<int>& histogram, 
                     int num_bins, const std::string& method);
void histogram_thrust(const std::vector<int>& input, std::vector<int>& histogram, int num_bins);
void histogram_cub(const std::vector<int>& input, std::vector<int>& histogram, int num_bins);

// Radix Sort algorithms
void test_radix_sort();
void radix_sort_custom(std::vector<unsigned int>& data, const std::string& method);
void radix_sort_cub(std::vector<unsigned int>& data);
void radix_sort_thrust(std::vector<unsigned int>& data);
void block_radix_sort_custom(std::vector<unsigned int>& data);

// BFS algorithms
void test_bfs();
struct Graph {
    int num_vertices;
    int num_edges;
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    
    Graph(int v, int e);
};
Graph generate_random_graph(int num_vertices, int avg_degree);
void bfs_custom(const Graph& g, int source, std::vector<int>& distances, const std::string& method);
void bfs_work_efficient(const Graph& g, int source, std::vector<int>& distances);
void bfs_direction_optimizing(const Graph& g, int source, std::vector<int>& distances);
void bfs_multi_source(const Graph& g, const std::vector<int>& sources, std::vector<int>& distances);
void bfs_cpu(const Graph& g, int source, std::vector<int>& distances);

// Scan algorithms  
void test_scan();
void scan_inclusive_custom(const std::vector<int>& input, std::vector<int>& output);
void scan_exclusive_custom(const std::vector<int>& input, std::vector<int>& output);
void segmented_scan_custom(const std::vector<int>& input, const std::vector<int>& flags, 
                          std::vector<int>& output);

// Stream Compaction algorithms
void test_compact();
void compact_custom(const std::vector<int>& input, std::vector<int>& output, 
                   std::function<bool(int)> predicate);
void compact_thrust(const std::vector<int>& input, std::vector<int>& output,
                   std::function<bool(int)> predicate);

// Matrix Multiplication algorithms
void test_matrix_multiply();
void matrix_multiply_naive(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<float>& C, int N);
void matrix_multiply_shared(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int N);
void matrix_multiply_cublas(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int N);

// Merge Sort algorithms
void test_merge_sort();
void merge_sort_custom(std::vector<int>& data);
void merge_sort_thrust(std::vector<int>& data);

// Convolution algorithms
void test_convolution();
void convolution_1d_custom(const std::vector<float>& signal, const std::vector<float>& kernel,
                          std::vector<float>& output);
void convolution_2d_custom(const std::vector<float>& image, const std::vector<float>& kernel,
                          std::vector<float>& output, int width, int height, int kernel_size);

// Utility functions
void print_gpu_info();
void demonstrate_cuda_graphs();
void test_memory_bandwidth();

// Performance analysis
template<typename KernelFunc>
void analyze_kernel_occupancy(KernelFunc kernel, const std::string& kernel_name);

// Algorithm performance comparison
struct PerformanceStats {
    std::string algorithm;
    float custom_time_ms;
    float thrust_time_ms;
    float cub_time_ms;
    float speedup_vs_thrust;
    float speedup_vs_cub;
    
    PerformanceStats(const std::string& alg);
};

extern std::vector<PerformanceStats> performance_results;
void print_performance_summary();