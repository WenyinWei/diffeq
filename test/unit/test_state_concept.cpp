#include <iostream>
#include <vector>
#include <array>
#include <deque>
#include <list>
#include <string>
#include <cassert>
#include "../include/core/concepts.hpp"

// Test helper macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            return false; \
        } else { \
            std::cout << "PASS: " << message << std::endl; \
        } \
    } while(0)

// Test classes for State concept testing
class ValidState {
public:
    using value_type = double;
    
    std::vector<double> data;
    
    ValidState(std::size_t size) : data(size, 0.0) {}
    
    std::size_t size() const { return data.size(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

class InvalidStateNoValueType {
public:
    std::vector<double> data;
    std::size_t size() const { return data.size(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

class InvalidStateNonArithmeticValueType {
public:
    using value_type = std::string;
    std::vector<std::string> data;
    std::size_t size() const { return data.size(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

class InvalidStateNoRandomAccessIterator {
public:
    using value_type = double;
    std::list<double> data;
    std::size_t size() const { return data.size(); }
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
};

// Test functions
bool test_time_type_concept() {
    std::cout << "\n=== Testing can_be_time Concept ===" << std::endl;
    
    // Valid time types
    TEST_ASSERT(can_be_time<int>, "int should satisfy can_be_time");
    TEST_ASSERT(can_be_time<float>, "float should satisfy can_be_time");
    TEST_ASSERT(can_be_time<double>, "double should satisfy can_be_time");
    TEST_ASSERT(can_be_time<long>, "long should satisfy can_be_time");
    TEST_ASSERT(can_be_time<short>, "short should satisfy can_be_time");
    TEST_ASSERT(can_be_time<unsigned int>, "unsigned int should satisfy can_be_time");
    TEST_ASSERT(can_be_time<bool>, "bool should satisfy can_be_time");
    TEST_ASSERT(can_be_time<char>, "char should satisfy can_be_time");
    TEST_ASSERT(can_be_time<long double>, "long double should satisfy can_be_time");
    
    // Invalid time types
    TEST_ASSERT(!can_be_time<std::string>, "std::string should NOT satisfy can_be_time");
    TEST_ASSERT(!can_be_time<std::vector<int>>, "std::vector<int> should NOT satisfy can_be_time");
    TEST_ASSERT(!can_be_time<void*>, "void* should NOT satisfy can_be_time");
    TEST_ASSERT(!can_be_time<int*>, "int* should NOT satisfy can_be_time");
    
    return true;
}

bool test_state_concept() {
    std::cout << "\n=== Testing system_state Concept ===" << std::endl;
    std::cout << "=== Testing system_state Concept ===" << std::endl;
    
    // Valid state types
    TEST_ASSERT(system_state<ValidState>, "ValidState should satisfy system_state");
    TEST_ASSERT(system_state<std::vector<double>>, "std::vector<double> should satisfy system_state");
    TEST_ASSERT(system_state<std::vector<float>>, "std::vector<float> should satisfy system_state");
    TEST_ASSERT(system_state<std::vector<int>>, "std::vector<int> should satisfy system_state");
    TEST_ASSERT((system_state<std::array<double, 10>>), "std::array<double, 10> should satisfy system_state");
    TEST_ASSERT(system_state<std::deque<float>>, "std::deque<float> should satisfy system_state");
    
    // Invalid state types
    TEST_ASSERT(!system_state<InvalidStateNoValueType>, 
               "InvalidStateNoValueType should NOT satisfy system_state");
    TEST_ASSERT(!system_state<InvalidStateNonArithmeticValueType>, 
               "InvalidStateNonArithmeticValueType should NOT satisfy system_state");
    TEST_ASSERT(!system_state<InvalidStateNoRandomAccessIterator>, 
               "InvalidStateNoRandomAccessIterator should NOT satisfy system_state");
    TEST_ASSERT(!system_state<int>, "int should NOT satisfy system_state");
    TEST_ASSERT(!system_state<std::string>, "std::string should NOT satisfy system_state");
    TEST_ASSERT(!system_state<std::vector<std::string>>, 
               "std::vector<std::string> should NOT satisfy system_state");
    TEST_ASSERT(!system_state<std::list<double>>, 
               "std::list<double> should NOT satisfy system_state");
    
    return true;
}

// Template function using system_state concept
template<system_state S>
void process_state(S& state) {
    for (std::size_t i = 0; i < state.size(); ++i) {
        auto it = state.begin();
        it[i] = static_cast<typename S::value_type>(i);
    }
}

bool test_concept_usage() {
    std::cout << "\n=== Testing system_state Concept Usage in Templates ===" << std::endl;
    
    // Test State concept in template
    ValidState state(5);
    process_state(state);
    bool correct_values = true;
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (std::abs(state.data[i] - static_cast<double>(i)) > 1e-10) {
            correct_values = false;
            break;
        }
    }
    TEST_ASSERT(correct_values, "process_state should set correct values in ValidState");
    
    std::vector<float> float_state(3);
    process_state(float_state);
    bool float_correct = true;
    for (std::size_t i = 0; i < float_state.size(); ++i) {
        if (std::abs(float_state[i] - static_cast<float>(i)) > 1e-6f) {
            float_correct = false;
            break;
        }
    }
    TEST_ASSERT(float_correct, "process_state should work with std::vector<float>");
    
    std::array<int, 4> int_state;
    process_state(int_state);
    bool int_correct = true;
    for (std::size_t i = 0; i < int_state.size(); ++i) {
        if (int_state[i] != static_cast<int>(i)) {
            int_correct = false;
            break;
        }
    }
    TEST_ASSERT(int_correct, "process_state should work with std::array<int, 4>");
    
    return true;
}

bool test_compile_time_checks() {
    std::cout << "\n=== Testing Compile-time system_state Concept Checks ===" << std::endl;
    
    // These are compile-time checks using static_assert
    static_assert(system_state<std::vector<double>>, 
                  "std::vector<double> should satisfy system_state at compile time");
    static_assert(system_state<ValidState>, 
                  "ValidState should satisfy system_state at compile time");
    static_assert((system_state<std::array<float, 5>>),
                  "std::array<float, 5> should satisfy system_state at compile time");
    static_assert(can_be_time<double>, 
                  "double should satisfy can_be_time at compile time");
    
    // Runtime checks for negative cases (some may not be well-formed for static_assert)
    constexpr bool string_not_state = !system_state<std::string>;
    TEST_ASSERT(string_not_state, "std::string should NOT satisfy system_state");
    
    constexpr bool list_not_state = !system_state<std::list<double>>;
    TEST_ASSERT(list_not_state, "std::list<double> should NOT satisfy system_state");
    
    std::cout << "PASS: All compile-time concept checks passed" << std::endl;
    return true;
}

int main() {
    std::cout << "Running Concepts Test Suite" << std::endl;
    std::cout << "============================" << std::endl;
    
    bool all_passed = true;
    
    try {
        all_passed &= test_time_type_concept();
        all_passed &= test_state_concept();
        all_passed &= test_concept_usage();
        all_passed &= test_compile_time_checks();
        
        std::cout << "\n=== Test Results ===" << std::endl;
        if (all_passed) {
            std::cout << "🎉 All tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ Some tests FAILED!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Test execution failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test execution failed with unknown exception" << std::endl;
        return 1;
    }
}
