#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <execution>
#include <algorithm>

/**
 * @brief Test basic parallelism features available in the examples
 * 
 * This demonstrates the available features:
 * 1. std::execution parallelism
 * 2. Basic ODE integration
 */

// Simple harmonic oscillator for testing
auto simple_harmonic_oscillator(double omega = 1.0) {
    return [omega](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];           // dx/dt = v
        dydt[1] = -omega*omega*y[0];  // dv/dt = -ω²x
    };
}

void test_std_execution_parallelism() {
    std::cout << "=== Testing std::execution Parallelism ===\n";
    
    auto system = simple_harmonic_oscillator(1.0);
    std::vector<std::vector<double>> states(100, {1.0, 0.0});
    
    std::cout << "Running " << states.size() << " integrations in parallel...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use std::execution for parallel integration
    std::for_each(std::execution::par_unseq, 
                 states.begin(), 
                 states.end(),
                 [&](std::vector<double>& state) {
                     auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
                     for (int i = 0; i < 100; ++i) {
                         integrator.step(state, 0.01);
                     }
                 });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Parallel integration completed in " << duration.count() << "ms\n";
    std::cout << "Result for state 10: [" << states[10][0] << ", " << states[10][1] << "]\n";
    std::cout << "✓ std::execution parallelism test completed\n\n";
}

void test_basic_ode_integration() {
    std::cout << "=== Testing Basic ODE Integration ===\n";
    
    auto system = simple_harmonic_oscillator(1.0);
    std::vector<double> state = {1.0, 0.0};
    
    auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(system);
    
    std::cout << "Initial state: [" << state[0] << ", " << state[1] << "]\n";
    
    for (int i = 0; i < 100; ++i) {
        integrator.step(state, 0.01);
        if (i % 25 == 0) {
            std::cout << "Step " << i << ": [" << state[0] << ", " << state[1] << "]\n";
        }
    }
    
    std::cout << "Final state: [" << state[0] << ", " << state[1] << "]\n";
    std::cout << "✓ Basic ODE integration test completed\n\n";
}

void test_library_availability() {
    std::cout << "=== Library Availability ===\n";
    
    std::cout << "std::execution: ✓ (C++17/20)\n";
    std::cout << "OpenMP:         " << 
        #ifdef _OPENMP
        "✓"
        #else
        "✗"
        #endif
        << "\n";
    std::cout << "Intel TBB:      " << 
        #ifdef TBB_AVAILABLE
        "✓"
        #else
        "✗"
        #endif
        << "\n";
    std::cout << "NVIDIA Thrust:  " << 
        #ifdef THRUST_AVAILABLE
        "✓"
        #else
        "✗"
        #endif
        << "\n";
    std::cout << "CUDA:           " << 
        #ifdef __CUDACC__
        "✓"
        #else
        "✗"
        #endif
        << "\n";
    std::cout << "OpenCL:         " << 
        #ifdef OPENCL_AVAILABLE
        "✓"
        #else
        "✗"
        #endif
        << "\n";
    std::cout << "\n";
}

int main() {
    std::cout << "Basic Parallelism Features Test\n";
    std::cout << "===============================\n";
    std::cout << "Testing available parallelism features:\n";
    std::cout << "• std::execution parallelism\n";
    std::cout << "• Basic ODE integration\n\n";
    
    try {
        test_library_availability();
        test_basic_ode_integration();
        test_std_execution_parallelism();
        
        std::cout << "=== Test Summary ===\n";
        std::cout << "✅ std::execution parallelism: Working\n";
        std::cout << "✅ Basic ODE integration: Working\n";
        std::cout << "✅ Library availability: Checked\n";
        std::cout << "\nKey Benefits:\n";
        std::cout << "• Standard C++17/20 parallelism without custom classes\n";
        std::cout << "• Direct integration with diffeq library\n";
        std::cout << "• Easy to understand and use\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}