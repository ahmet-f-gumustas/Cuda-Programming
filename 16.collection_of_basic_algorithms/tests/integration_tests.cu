// ==================== tests/integration_tests.cu ====================
#include "../include/algorithms.h"

// Integration test that combines multiple algorithms
void test_algorithm_pipeline() {
    std::cout << "\n=== INTEGRATION TEST: Algorithm Pipeline ===" << std::endl;
    
    const int n = 100000;
    auto input = generate_random_data<int>(n, 1, 100);
    
    // Step 1: Prefix sum
    std::vector<int> prefix_result;
    prefix_sum_custom(input, prefix_result, "shared");
    
    // Step 2: Filter even numbers (stream compaction)
    std::vector<int> filtered_result;
    compact_custom(prefix_result, filtered_result, [](int x) { return x % 2 == 0; });
    
    // Step 3: Reduce to single value
    int final_result = reduce_custom(filtered_result, "shared");
    
    std::cout << "Pipeline results:" << std::endl;
    std::cout << "  Input size: " << input.size() << std::endl;
    std::cout << "  After prefix sum: " << prefix_result.size() << std::endl;
    std::cout << "  After filtering: " << filtered_result.size() << std::endl;
    std::cout << "  Final result: " << final_result << std::endl;
    
    // Verify pipeline makes sense
    assert(prefix_result.size() == input.size());
    assert(filtered_result.size() <= prefix_result.size());
    assert(final_result > 0);
    
    std::cout << "✅ Algorithm pipeline test passed!" << std::endl;
}