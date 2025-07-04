#include <execution/parallelism_facade_clean.hpp>
#include <execution/parallel_builder.hpp>
#include <execution/hardware_support.hpp>
#include <iostream>
#include <cassert>
#include <vector>

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
        
        facade.parallel_for_each(numbers.begin(), numbers.end(), [](int& num) {
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
        
        // Test 5: Async execution
        std::cout << "\n5. Testing async execution...\n";
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
        
        // Test 6: Different hardware targets
        std::cout << "\n6. Testing different hardware targets...\n";
        
        std::vector<diffeq::execution::HardwareTarget> targets = {
            diffeq::execution::HardwareTarget::CPU_Sequential,
            diffeq::execution::HardwareTarget::CPU_ThreadPool,
            diffeq::execution::HardwareTarget::GPU_CUDA,
            diffeq::execution::HardwareTarget::GPU_OpenCL,
            diffeq::execution::HardwareTarget::FPGA_HLS
        };
        
        for (auto target : targets) {
            std::string target_name;
            switch (target) {
                case diffeq::execution::HardwareTarget::CPU_Sequential: target_name = "CPU Sequential"; break;
                case diffeq::execution::HardwareTarget::CPU_ThreadPool: target_name = "CPU Thread Pool"; break;
                case diffeq::execution::HardwareTarget::GPU_CUDA: target_name = "GPU CUDA"; break;
                case diffeq::execution::HardwareTarget::GPU_OpenCL: target_name = "GPU OpenCL"; break;
                case diffeq::execution::HardwareTarget::FPGA_HLS: target_name = "FPGA HLS"; break;
                default: target_name = "Unknown"; break;
            }
            
            std::cout << "Testing " << target_name << "... ";
            
            diffeq::execution::ParallelConfig config;
            config.target = target;
            config.max_workers = 4;
            
            try {
                diffeq::execution::ParallelismFacade test_facade(config);
                
                if (test_facade.is_target_available(target)) {
                    std::cout << "Available (concurrency: " << test_facade.get_max_concurrency() << ")\n";
                } else {
                    std::cout << "Not available\n";
                }
            } catch (const std::exception& e) {
                std::cout << "Error: " << e.what() << "\n";
            }
        }
        
        // Test 7: Preset configurations
        std::cout << "\n7. Testing preset configurations...\n";
        
        auto robotics_facade = diffeq::execution::presets::robotics_control().build();
        std::cout << "Robotics control preset: " << robotics_facade->get_max_concurrency() << " max concurrency\n";
        
        auto research_facade = diffeq::execution::presets::stochastic_research().build();
        std::cout << "Stochastic research preset: " << research_facade->get_max_concurrency() << " max concurrency\n";
        
        auto monte_carlo_facade = diffeq::execution::presets::monte_carlo().build();
        std::cout << "Monte Carlo preset: " << monte_carlo_facade->get_max_concurrency() << " max concurrency\n";
        
        auto realtime_facade = diffeq::execution::presets::real_time_systems().build();
        std::cout << "Real-time systems preset: " << realtime_facade->get_max_concurrency() << " max concurrency\n";
        
        std::cout << "\n✅ All basic parallelism tests passed!\n";
        
        // Test 8: Pattern functions
        std::cout << "\n8. Testing parallel patterns...\n";
        
        std::vector<int> input = {1, 2, 3, 4, 5};
        auto squared = diffeq::execution::patterns::parallel_map(
            input.begin(), input.end(), [](int x) { return x * x; });
        
        std::cout << "Parallel map result: ";
        for (int val : squared) {
            std::cout << val << " ";
        }
        std::cout << "\n";
        
        int sum = diffeq::execution::patterns::parallel_reduce(
            input.begin(), input.end(), 0, std::plus<int>());
        std::cout << "Parallel reduce (sum): " << sum << "\n";
        
        std::cout << "\n✅ All parallelism features tested successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}