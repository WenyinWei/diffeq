#include <iostream>
#include <vector>
#include <array>
#include <deque>
#include "core/concepts.hpp"

// Example function that works with any system_state-satisfying type
template<system_state S>
void print_state_info(const S& state) {
    std::cout << "State type: " << typeid(S).name() << std::endl;
    std::cout << "Size: " << state.size() << std::endl;
    std::cout << "Values: ";
    for (const auto& value : state) {
        std::cout << value << " ";
    }
    std::cout << std::endl << std::endl;
}

// Example function that modifies a state
template<system_state S>
void initialize_state(S& state, typename S::value_type initial_value = {}) {
    for (std::size_t i = 0; i < state.size(); ++i) {
        auto it = state.begin();
        it[i] = initial_value + static_cast<typename S::value_type>(i);
    }
}

int main() {
    std::cout << "State Concept Usage Examples" << std::endl;
    std::cout << "============================" << std::endl << std::endl;
    
    // Example 1: std::vector<double> (common for numerical computations)
    std::vector<double> vector_state(5);
    initialize_state(vector_state, 1.0);
    std::cout << "Example 1 - Vector State:" << std::endl;
    print_state_info(vector_state);
    
    // Example 2: std::array<float, 4> (fixed-size, good for small states)
    std::array<float, 4> array_state;
    initialize_state(array_state, 2.5f);
    std::cout << "Example 2 - Array State:" << std::endl;
    print_state_info(array_state);
    
    // Example 3: std::deque<int> (good for dynamic resizing)
    std::deque<int> deque_state(3);
    initialize_state(deque_state, 10);
    std::cout << "Example 3 - Deque State:" << std::endl;
    print_state_info(deque_state);
    
    // Example 4: Custom state class
    class CustomState {
    public:
        using value_type = double;
        
        std::vector<double> data;
        std::string name;
        
        CustomState(const std::string& n, std::size_t size) 
            : data(size, 0.0), name(n) {}
        
        std::size_t size() const { return data.size(); }
        auto begin() { return data.begin(); }
        auto end() { return data.end(); }
        auto begin() const { return data.begin(); }
        auto end() const { return data.end(); }
        
        const std::string& get_name() const { return name; }
    };
    
    CustomState custom_state("MyState", 6);
    initialize_state(custom_state, 0.5);
    std::cout << "Example 4 - Custom State (name: " << custom_state.get_name() << "):" << std::endl;
    print_state_info(custom_state);
    
    return 0;
}
