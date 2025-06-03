// ==================== tests/test_main.cu ====================
#include "../include/algorithms.h"
#include "../include/performance.h"

int main(int argc, char* argv[]) {
    std::cout << "=== CUDA PARALLEL ALGORITHMS TEST SUITE ===" << std::endl;
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    print_gpu_info();
    
    bool all_tests_passed = true;
    
    try {
        // Test all algorithms
        std::cout << "\n🧪 Running algorithm tests...\n" << std::endl;
        
        test_prefix_sum();
        test_reduce();
        test_histogram();
        test_radix_sort();
        test_bfs();
        test_scan();
        test_compact();
        test_matrix_multiply();
        test_merge_sort();
        test_convolution();
        
        std::cout << "\n✅ All algorithm tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        all_tests_passed = false;
    }
    
    // Performance summary
    print_performance_summary();
    
    // Additional tests
    std::cout << "\n🚀 Running additional tests...\n" << std::endl;
    test_memory_bandwidth();
    demonstrate_cuda_graphs();
    
    if (all_tests_passed) {
        std::cout << "\n🎉 All tests PASSED! 🎉" << std::endl;
        return 0;
    } else {
        std::cout << "\n💥 Some tests FAILED! 💥" << std::endl;
        return 1;
    }
}