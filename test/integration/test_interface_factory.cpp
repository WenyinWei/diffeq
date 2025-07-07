#include <iostream>
#include <vector>
#include <memory>
#include <interfaces/integration_interface.hpp>

int main() {
    std::cout << "Testing interface factory function..." << std::endl;
    
    try {
        // Test the factory function that's failing
        std::cout << "About to call make_integration_interface..." << std::endl;
        auto interface = diffeq::interfaces::make_integration_interface<std::vector<double>, double>();
        std::cout << "✓ Factory function succeeded" << std::endl;
        
        if (interface) {
            std::cout << "✓ Interface pointer is valid" << std::endl;
        } else {
            std::cout << "✗ Interface pointer is null" << std::endl;
            return 1;
        }
        
        std::cout << "All factory tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
} 