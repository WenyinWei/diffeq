#include <iostream>
#include <vector>
#include <memory>
#include <signal/signal_processor.hpp>

// Try to include the interface header directly
#include <interfaces/integration_interface.hpp>

int main() {
    std::cout << "Testing interface basic instantiation..." << std::endl;
    
    try {
        // Test 1: Try to create the template type explicitly
        using InterfaceType = diffeq::interfaces::IntegrationInterface<std::vector<double>, double>;
        std::cout << "✓ Template type created successfully" << std::endl;
        
        // Test 2: Try to construct the interface with explicit constructor
        auto signal_proc = diffeq::signal::make_signal_processor<std::vector<double>>();
        InterfaceType interface(signal_proc);
        std::cout << "✓ Interface constructed successfully" << std::endl;
        
        std::cout << "All basic interface tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
} 