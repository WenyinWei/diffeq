#include <execution/parallel.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <chrono>

int main() {
    std::cout << "Testing Simplified Parallelism Interface for diffeq library\n";
    std::cout << "=========================================================\n";
    
    try {
        // Test 1: Simple global parallel execution
        std::cout << "\n1. Testing simple global parallel execution...\n";
        
        std::vector<int> data(1000);
        std::iota(data.begin(), data.end(), 1); // Fill with 1, 2, 3, ..., 1000
        
        // Square each element in parallel - super simple!
        diffeq::execution::parallel_for_each(data, [](int& x) {
            x = x * x;
        });
        
        // Verify result
        assert(data[0] == 1);
        assert(data[1] == 4);
        assert(data[2] == 9);
        std::cout << "✓ Global parallel_for_each works correctly\n";
        
        // Test 2: Custom parallel instance
        std::cout << "\n2. Testing custom parallel instance...\n";
        
        auto parallel = diffeq::execution::Parallel(4); // Use 4 workers
        std::cout << "Created parallel executor with " << parallel.worker_count() << " workers\n";
        
        std::vector<double> values(100, 1.0);
        parallel.for_each(values, [](double& val) {
            val = val * 2.0 + 1.0; // Transform: x -> 2x + 1
        });
        
        // All values should be 3.0 (2*1.0 + 1.0)
        assert(values[0] == 3.0);
        assert(values[50] == 3.0);
        std::cout << "✓ Custom parallel instance works correctly\n";
        
        // Test 3: Async execution
        std::cout << "\n3. Testing async execution...\n";
        
        auto future = diffeq::execution::parallel_async([]() {
            // Simulate some computation
            int sum = 0;
            for (int i = 0; i < 1000; ++i) {
                sum += i * i;
            }
            return sum;
        });
        
        auto result = future.get();
        int expected = 332833500; // Sum of squares from 0 to 999
        assert(result == expected);
        std::cout << "✓ Async execution works correctly, result: " << result << "\n";
        
        // Test 4: Hardware detection and switching
        std::cout << "\n4. Testing hardware detection...\n";
        
        auto& global_parallel = diffeq::execution::parallel();
        std::cout << "GPU available: " << (global_parallel.gpu_available() ? "Yes" : "No") << "\n";
        
        // Try to use GPU if available, otherwise stick with CPU
        if (global_parallel.gpu_available()) {
            global_parallel.use_gpu();
            std::cout << "Switched to GPU execution\n";
        } else {
            global_parallel.use_cpu();
            std::cout << "Using CPU execution\n";
        }
        
        // Test 5: Configuration functions
        std::cout << "\n5. Testing global configuration...\n";
        
        diffeq::execution::set_parallel_workers(8);
        std::cout << "Set global workers to 8, current count: " << global_parallel.worker_count() << "\n";
        
        // Test 6: Real-world example - ODE integration states
        std::cout << "\n6. Testing real-world ODE integration scenario...\n";
        
        // Simulate multiple initial conditions for an ODE system
        struct ODEState {
            std::vector<double> y;
            double t;
            
            ODEState(const std::vector<double>& initial, double time = 0.0) 
                : y(initial), t(time) {}
        };
        
        std::vector<ODEState> initial_conditions;
        for (int i = 0; i < 100; ++i) {
            initial_conditions.emplace_back(
                std::vector<double>{static_cast<double>(i), static_cast<double>(i*2)}, 
                0.0
            );
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Parallel integration step for all initial conditions
        diffeq::execution::parallel_for_each(initial_conditions, [](ODEState& state) {
            // Simulate a simple ODE integration step: dy/dt = -0.1 * y
            double dt = 0.01;
            for (auto& component : state.y) {
                component *= (1.0 - 0.1 * dt); // Simple Euler step
            }
            state.t += dt;
        });
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "✓ Parallel ODE step completed for " << initial_conditions.size() 
                  << " initial conditions in " << duration.count() << " μs\n";
        
        std::cout << "\n✅ All simplified parallelism tests passed!\n";
        std::cout << "\nUsage Summary:\n";
        std::cout << "=============\n";
        std::cout << "Simple: diffeq::execution::parallel_for_each(container, lambda)\n";
        std::cout << "Async:  auto future = diffeq::execution::parallel_async(lambda)\n";
        std::cout << "Custom: auto p = diffeq::execution::Parallel(workers); p.for_each(...)\n";
        std::cout << "Config: diffeq::execution::set_parallel_workers(N)\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}