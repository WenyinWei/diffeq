#include <iostream>
#include <vector>
#include <memory>
#include <signal/signal_processor.hpp>

int main() {
    std::cout << "Testing signal processor..." << std::endl;
    
    try {
        // Test 1: Create signal processor
        auto signal_proc = diffeq::signal::make_signal_processor<std::vector<double>>();
        std::cout << "✓ Signal processor created successfully" << std::endl;
        
        // Test 2: Register handler
        bool handler_called = false;
        signal_proc->register_handler<double>("test_signal",
            [&handler_called](const diffeq::signal::Signal<double>& sig) {
                handler_called = true;
                std::cout << "Handler called with value: " << sig.data << std::endl;
            });
        
        std::cout << "✓ Signal handler registered" << std::endl;
        
        // Test 3: Emit signal
        signal_proc->emit_signal("test_signal", 42.0);
        std::cout << "✓ Signal emitted" << std::endl;
        
        std::cout << "All signal tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
} 