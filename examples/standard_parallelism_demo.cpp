#include <examples/standard_parallelism.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

/**
 * @brief Demonstration of using standard parallelism libraries with diffeq
 * 
 * This example shows the user's requested approach: using standard libraries
 * instead of custom parallel classes, with flexibility beyond just initial conditions.
 */

// Simple harmonic oscillator system for demonstration
struct HarmonicOscillator {
    double omega = 1.0;  // Frequency parameter - can be varied!
    
    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double t) const {
        dydt[0] = y[1];           // dx/dt = v
        dydt[1] = -omega*omega*y[0];  // dv/dt = -ω²x
    }
    
    // Overload for parameter sweep
    void operator()(const std::vector<double>& y, std::vector<double>& dydt, double t, double param_omega) const {
        dydt[0] = y[1];
        dydt[1] = -param_omega*param_omega*y[0];
    }
};

void print_timing(const std::string& method, std::chrono::microseconds duration, size_t operations) {
    std::cout << std::setw(20) << method << ": " 
              << std::setw(8) << duration.count() << " μs"
              << " (" << operations << " operations, "
              << std::fixed << std::setprecision(2)
              << (duration.count() / static_cast<double>(operations)) << " μs/op)"
              << std::endl;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "diffeq Standard Parallelism Integration Examples\n";
    std::cout << "=================================================================\n";
    std::cout << "Demonstrating how to use standard libraries instead of custom parallel classes\n\n";
    
    // Check what's available
    std::cout << "Available Parallelism Libraries:\n";
    std::cout << "- std::execution: " << (diffeq::examples::availability::std_execution_available() ? "✓" : "✗") << "\n";
    std::cout << "- OpenMP:         " << (diffeq::examples::availability::openmp_available() ? "✓" : "✗") << "\n";
    std::cout << "- Intel TBB:      " << (diffeq::examples::availability::tbb_available() ? "✓" : "✗") << "\n";
    std::cout << "- NVIDIA Thrust:  " << (diffeq::examples::availability::thrust_available() ? "✓" : "✗") << "\n\n";
    
    // Test problem setup
    const int num_conditions = 1000;
    const double dt = 0.01;
    const int steps = 100;
    
    HarmonicOscillator system;
    
    // ================================================================
    // Example 1: Multiple Initial Conditions with std::execution
    // ================================================================
    std::cout << "1. Multiple Initial Conditions using std::execution\n";
    std::cout << "   ================================================\n";
    
    std::vector<std::vector<double>> initial_conditions(num_conditions);
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i) * 0.1, 0.0}; // x₀ = i*0.1, v₀ = 0
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    diffeq::examples::StandardParallelODE<std::vector<double>, double>::integrate_multiple_conditions(
        system, initial_conditions, dt, steps
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("std::execution", duration, num_conditions);
    std::cout << "   Final position of condition 100: " << initial_conditions[100][0] << "\n\n";
    
    // ================================================================
    // Example 2: Parameter Sweep - Beyond Initial Conditions!
    // ================================================================
    std::cout << "2. Parameter Sweep using std::execution (flexibility beyond initial conditions)\n";
    std::cout << "   ===========================================================================\n";
    
    // Different frequency parameters for harmonic oscillator
    std::vector<double> omega_values;
    for (int i = 1; i <= 100; ++i) {
        omega_values.push_back(i * 0.1); // ω = 0.1, 0.2, ..., 10.0
    }
    
    std::vector<std::vector<double>> parameter_results;
    std::vector<double> initial_state = {1.0, 0.0}; // Start from x=1, v=0
    
    start = std::chrono::high_resolution_clock::now();
    diffeq::examples::StandardParallelODE<std::vector<double>, double>::parameter_sweep(
        [](const std::vector<double>& y, std::vector<double>& dydt, double t, double omega) {
            dydt[0] = y[1];
            dydt[1] = -omega*omega*y[0];
        },
        initial_state, omega_values, parameter_results, dt, steps
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("Parameter Sweep", duration, omega_values.size());
    std::cout << "   Result for ω=1.0: x=" << parameter_results[9][0] << ", v=" << parameter_results[9][1] << "\n\n";
    
#ifdef _OPENMP
    // ================================================================
    // Example 3: OpenMP Parallelism
    // ================================================================
    std::cout << "3. OpenMP Parallel Integration\n";
    std::cout << "   ============================\n";
    
    // Reset initial conditions
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i) * 0.1, 0.0};
    }
    
    start = std::chrono::high_resolution_clock::now();
    diffeq::examples::OpenMPParallelODE<std::vector<double>, double>::integrate_openmp(
        system, initial_conditions, dt, steps
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("OpenMP", duration, num_conditions);
    
    // Demonstrate multiple integrators running simultaneously
    std::vector<std::vector<double>> rk4_results, euler_results;
    start = std::chrono::high_resolution_clock::now();
    diffeq::examples::OpenMPParallelODE<std::vector<double>, double>::multi_integrator_comparison(
        system, {1.0, 0.0}, dt, steps/10, rk4_results, euler_results
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("Multi-Integrator", duration, rk4_results.size() + euler_results.size());
    std::cout << "   RK4 vs Euler difference: " 
              << std::abs(rk4_results[0][0] - euler_results[0][0]) << "\n\n";
#endif

#ifdef TBB_AVAILABLE
    // ================================================================
    // Example 4: Intel TBB Parallelism
    // ================================================================
    std::cout << "4. Intel TBB Parallel Integration\n";
    std::cout << "   ===============================\n";
    
    // Reset initial conditions
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i) * 0.1, 0.0};
    }
    
    start = std::chrono::high_resolution_clock::now();
    diffeq::examples::TBBParallelODE<std::vector<double>, double>::integrate_tbb(
        system, initial_conditions, dt, steps
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("TBB", duration, num_conditions);
    
    // Blocked execution for memory efficiency
    for (int i = 0; i < num_conditions; ++i) {
        initial_conditions[i] = {static_cast<double>(i) * 0.1, 0.0};
    }
    
    start = std::chrono::high_resolution_clock::now();
    diffeq::examples::TBBParallelODE<std::vector<double>, double>::integrate_blocked(
        system, initial_conditions, dt, steps, 100
    );
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    print_timing("TBB Blocked", duration, num_conditions);
    std::cout << "\n";
#endif

#ifdef THRUST_AVAILABLE
    // ================================================================
    // Example 5: GPU with NVIDIA Thrust (NO custom kernels!)
    // ================================================================
    std::cout << "5. GPU Integration using NVIDIA Thrust (NO custom kernels!)\n";
    std::cout << "   ========================================================\n";
    
    if (diffeq::examples::ThrustGPUODE<std::vector<double>, double>::gpu_available()) {
        std::cout << "GPU detected! Demonstrating GPU ODE integration without custom kernels...\n";
        
        // Smaller problem size for GPU demo
        std::vector<std::vector<double>> gpu_conditions(100);
        for (int i = 0; i < 100; ++i) {
            gpu_conditions[i] = {static_cast<double>(i) * 0.1, 0.0};
        }
        
        start = std::chrono::high_resolution_clock::now();
        diffeq::examples::ThrustGPUODE<std::vector<double>, double>::integrate_gpu(
            system, gpu_conditions, dt, steps
        );
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        print_timing("GPU (Thrust)", duration, gpu_conditions.size());
        std::cout << "   GPU result: " << gpu_conditions[50][0] << "\n";
    } else {
        std::cout << "No GPU available - skipping GPU demonstration\n";
    }
    std::cout << "\n";
#endif

    // ================================================================
    // Summary and Usage Recommendations
    // ================================================================
    std::cout << "Usage Recommendations:\n";
    std::cout << "======================\n";
    std::cout << "• For simple parallel loops: Use std::for_each with std::execution::par\n";
    std::cout << "• For CPU-intensive work: Use OpenMP pragmas\n";
    std::cout << "• For complex task scheduling: Use Intel TBB\n";
    std::cout << "• For GPU acceleration: Use NVIDIA Thrust (no custom kernels needed!)\n";
    std::cout << "• Mix and match based on your specific needs\n";
    std::cout << "• No custom parallel classes needed - standard libraries are sufficient!\n\n";
    
    std::cout << "Key Advantages:\n";
    std::cout << "• ✓ Use proven, optimized standard libraries\n";
    std::cout << "• ✓ No custom 'facade' classes to learn\n";
    std::cout << "• ✓ Flexibility beyond just initial conditions\n";
    std::cout << "• ✓ GPU support without writing CUDA kernels\n";
    std::cout << "• ✓ Hardware detection using standard library functions\n";
    std::cout << "• ✓ Choose the right tool for each specific use case\n";
    
    return 0;
}