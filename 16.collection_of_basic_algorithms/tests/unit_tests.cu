// ==================== tests/unit_tests.cu ====================
#include "../include/algorithms.h"
#include <cassert>

// Unit test utilities
#define ASSERT_TRUE(condition) do { \
    if (!(condition)) { \
        std::cerr << "Assertion failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return false; \
    } \
} while(0)

#define ASSERT_FLOAT_EQ(a, b, tolerance) do { \
    if (std::abs((a) - (b)) > (tolerance)) { \
        std::cerr << "Float assertion failed: " << (a) << " != " << (b) << " (tolerance: " << (tolerance) << ")" << std::endl; \
        return false; \
    } \
} while(0)

// Test small prefix sum
bool test_small_prefix_sum() {
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> expected = {1, 3, 6, 10, 15};
    std::vector<int> output;
    
    prefix_sum_custom(input, output, "shared");
    
    ASSERT_TRUE(output.size() == expected.size());
    for (size_t i = 0; i < output.size(); ++i) {
        ASSERT_TRUE(output[i] == expected[i]);
    }
    
    return true;
}

// Test small reduce
bool test_small_reduce() {
    std::vector<int> input = {1, 2, 3, 4, 5};
    int expected = 15;
    
    int result = reduce_custom(input, "shared");
    ASSERT_TRUE(result == expected);
    
    return true;
}

// Test edge cases
bool test_edge_cases() {
    // Empty array
    std::vector<int> empty_input;
    std::vector<int> empty_output;
    
    // Single element
    std::vector<int> single_input = {42};
    std::vector<int> single_output;
    prefix_sum_custom(single_input, single_output, "shared");
    ASSERT_TRUE(single_output.size() == 1);
    ASSERT_TRUE(single_output[0] == 42);
    
    // Two elements
    std::vector<int> two_input = {10, 20};
    std::vector<int> two_output;
    prefix_sum_custom(two_input, two_output, "shared");
    ASSERT_TRUE(two_output.size() == 2);
    ASSERT_TRUE(two_output[0] == 10);
    ASSERT_TRUE(two_output[1] == 30);
    
    return true;
}

// Run all unit tests
void run_unit_tests() {
    std::cout << "\n=== UNIT TESTS ===" << std::endl;
    
    std::vector<std::pair<std::string, bool(*)()>> tests = {
        {"Small Prefix Sum", test_small_prefix_sum},
        {"Small Reduce", test_small_reduce},
        {"Edge Cases", test_edge_cases}
    };
    
    int passed = 0;
    int total = tests.size();
    
    for (const auto& test : tests) {
        std::cout << "Running " << test.first << "... ";
        if (test.second()) {
            std::cout << "✅ PASSED" << std::endl;
            passed++;
        } else {
            std::cout << "❌ FAILED" << std::endl;
        }
    }
    
    std::cout << "\nUnit Test Results: " << passed << "/" << total << " passed" << std::endl;
}