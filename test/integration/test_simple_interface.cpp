#include <iostream>
#include <vector>
#include <memory>
#include <interfaces/integration_interface.hpp>
#include <signal/signal_processor.hpp>

using namespace diffeq;

int main() {
    std::cout << "Testing simple interface creation..." << std::endl;
    
    try {
        // Test 1: Simple interface creation
        auto interface = interfaces::make_integration_interface<std::vector<double>, double>();
        std::cout << "✓ Interface created successfully" << std::endl;
        
        // Test 2: Simple handler registration
        bool handler_called = false;
        auto handler = [&handler_called](const double& value, std::vector<double>& state, double t) {
            handler_called = true;
            std::cout << "Handler called with value: " << value << std::endl;
        };
        
        interface->register_signal_influence<double>("test_signal",
            interfaces::IntegrationInterface<std::vector<double>, double>::InfluenceMode::DISCRETE_EVENT,
            handler);
        
        std::cout << "✓ Signal influence registered" << std::endl;
        
        // Test 3: Signal processor access
        auto signal_proc = interface->get_signal_processor();
        signal_proc->emit_signal("test_signal", 42.0);
        
        std::cout << "✓ Signal emitted" << std::endl;
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
} 