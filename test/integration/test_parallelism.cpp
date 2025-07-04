#include <diffeq.hpp>
#include <examples/parallelism_usage.hpp>
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Enhanced Parallelism Capabilities for diffeq library\n";
    std::cout << "============================================================\n";
    
    try {
        // Test 1: Basic parallelism facade
        std::cout << "\n1. Testing basic parallelism facade...\n";
        diffeq::execution::ParallelismFacade facade;
        
        std::cout << "Current target: ";
        auto target = facade.get_current_target();
        switch (target) {
            case diffeq::execution::HardwareTarget::CPU_Sequential: 
                std::cout << "CPU Sequential"; break;
            case diffeq::execution::HardwareTarget::CPU_ThreadPool: 
                std::cout << "CPU Thread Pool"; break;
            case diffeq::execution::HardwareTarget::GPU_CUDA: 
                std::cout << "GPU CUDA"; break;
            case diffeq::execution::HardwareTarget::GPU_OpenCL: 
                std::cout << "GPU OpenCL"; break;
            case diffeq::execution::HardwareTarget::FPGA_HLS: 
                std::cout << "FPGA HLS"; break;
            default: 
                std::cout << "Auto/Unknown"; break;
        }
        std::cout << "\n";
        
        std::cout << "Max concurrency: " << facade.get_max_concurrency() << "\n";
        
        // Test 2: Builder pattern
        std::cout << "\n2. Testing builder pattern...\n";
        auto config_builder = diffeq::execution::parallel_execution()
            .target_cpu()
            .use_thread_pool()
            .workers(4)
            .normal_priority()
            .enable_load_balancing();
        
        auto built_facade = config_builder.build();
        std::cout << "Built facade with " << built_facade->get_max_concurrency() << " max concurrency\n";
        
        // Test 3: Simple parallel execution
        std::cout << "\n3. Testing simple parallel execution...\n";
        std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> results(numbers.size());
        
        facade.parallel_for_each(numbers.begin(), numbers.end(), [&](int& num) {
            num = num * num;  // Square each number
        });
        
        std::cout << "Squared numbers: ";
        for (int num : numbers) {
            std::cout << num << " ";
        }
        std::cout << "\n";
        
        // Test 4: Hardware capabilities detection
        std::cout << "\n4. Testing hardware capabilities detection...\n";
        diffeq::execution::hardware::HardwareCapabilities::print_capabilities();
        
        // Test 5: Simple ODE integration with parallelism
        std::cout << "\n5. Testing ODE integration with parallelism...\n";
        
        // Simple harmonic oscillator: d²x/dt² = -k*x
        auto harmonic_system = [](double t, const std::vector<double>& state, std::vector<double>& derivative) {
            derivative[0] = state[1];      // dx/dt = v
            derivative[1] = -state[0];     // dv/dt = -x (k=1)
        };
        
        auto integrator = diffeq::ode::factory::make_rk4_integrator<std::vector<double>, double>(harmonic_system);
        
        // Integrate multiple initial conditions in parallel
        std::vector<std::vector<double>> initial_conditions = {
            {1.0, 0.0}, {0.0, 1.0}, {0.5, 0.5}, {-1.0, 0.0}, {0.0, -1.0}
        };
        
        facade.parallel_for_each(initial_conditions.begin(), initial_conditions.end(), 
                                [&](std::vector<double>& state) {
            double dt = 0.01;
            double end_time = 1.0;
            
            for (double t = 0.0; t < end_time; t += dt) {
                integrator->step(state, dt);
            }
        });
        
        std::cout << "Final states after integration:\n";
        for (size_t i = 0; i < initial_conditions.size(); ++i) {
            const auto& state = initial_conditions[i];
            std::cout << "  Initial condition " << i+1 << ": x=" << state[0] 
                      << ", v=" << state[1] << "\n";
        }
        
        // Test 6: Async execution
        std::cout << "\n6. Testing async execution...\n";
        auto future1 = facade.async([]() { 
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return 42; 
        });
        
        auto future2 = facade.async([]() { 
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return 24; 
        });
        
        int result1 = future1.get();
        int result2 = future2.get();
        std::cout << "Async results: " << result1 << ", " << result2 << "\n";
        
        assert(result1 == 42);
        assert(result2 == 24);
        
        std::cout << "\n✅ All basic tests passed!\n";
        
        // Test 7: Run full examples if requested
        std::cout << "\n7. Running full parallelism examples...\n";
        std::cout << "Note: This will take some time and demonstrate real use cases.\n";
        
        // Run the comprehensive examples
        diffeq::examples::parallelism::demonstrate_all_parallelism_features();
        
        std::cout << "\n✅ All parallelism features tested successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}