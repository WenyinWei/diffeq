#include <iostream>
#include <vector>
#include <array>
#include <core/concepts.hpp>

int main() {
    std::cout << "Testing core concepts..." << std::endl;
    
    // Test concepts with static_assert (concepts are defined in global namespace)
    static_assert(system_state<std::vector<double>>, "vector<double> should satisfy system_state");
    static_assert(system_state<std::array<double, 6>>, "array<double, 6> should satisfy system_state");
    static_assert(can_be_time<double>, "double should satisfy can_be_time");
    static_assert(can_be_time<float>, "float should satisfy can_be_time");
    
    // Test that invalid types are rejected
    static_assert(!system_state<int>, "int should not satisfy system_state");
    static_assert(!system_state<std::string>, "string should not satisfy system_state");
    static_assert(!can_be_time<std::string>, "string should not satisfy can_be_time");
    
    std::cout << "✓ All concept tests passed!" << std::endl;
    return 0;
} 