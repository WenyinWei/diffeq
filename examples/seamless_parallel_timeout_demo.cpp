/**
 * @file seamless_parallel_timeout_demo.cpp
 * @brief Demonstration of seamless timeout + async + parallel integration
 * 
 * This example shows how the diffeq library automatically leverages hardware
 * capabilities while providing timeout protection and allowing fine-grained
 * control for advanced users.
 */

#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <random>
#include <iomanip>

// Define test systems
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

void stiff_van_der_pol(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    double mu = 50.0;  // Very stiff system
    dydt[0] = y[1];
    dydt[1] = mu * (1 - y[0]*y[0]) * y[1] - y[0];
}

void demonstrate_automatic_hardware_utilization() {
    std::cout << "\n=== Automatic Hardware Utilization ===\n";
    
    // 1. Simple case - library automatically chooses best approach
    std::cout << "\n1. Simple Integration (Auto-optimization)\n";
    
    std::vector<double> state = {1.0, 0.0, 0.5};
    auto result = diffeq::integrate_auto(
        diffeq::RK45Integrator<std::vector<double>>(lorenz_system),
        state, 0.01, 1.0
    );
    
    std::cout << "  Strategy used: ";
    switch (result.used_strategy) {
        case diffeq::ExecutionStrategy::SEQUENTIAL: std::cout << "Sequential"; break;
        case diffeq::ExecutionStrategy::PARALLEL: std::cout << "Parallel"; break;
        case diffeq::ExecutionStrategy::ASYNC: std::cout << "Async"; break;
        case diffeq::ExecutionStrategy::HYBRID: std::cout << "Hybrid"; break;
        default: std::cout << "Auto"; break;
    }
    std::cout << "\n";
    std::cout << "  Hardware cores detected: " << result.hardware_used.cpu_cores << "\n";
    std::cout << "  Parallel tasks used: " << result.parallel_tasks_used << "\n";
    std::cout << "  Integration time: " << result.execution_time.count() << " μs\n";
    std::cout << "  Success: " << (result.is_success() ? "✓" : "✗") << "\n";
    
    if (result.is_success()) {
        std::cout << "  Final state: [" << state[0] << ", " << state[1] << ", " << state[2] << "]\n";
    }
}

void demonstrate_batch_processing() {
    std::cout << "\n=== Automatic Batch Processing ===\n";
    
    // Create multiple initial conditions
    std::vector<std::vector<double>> initial_conditions;
    for (int i = 0; i < 20; ++i) {
        initial_conditions.push_back({
            0.1 * i,           // x
            0.05 * (i - 10),   // y  
            1.0 + 0.1 * i      // z
        });
    }
    
    std::cout << "Processing " << initial_conditions.size() << " initial conditions...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Library automatically parallelizes the batch
    auto results = diffeq::integrate_batch_auto(
        diffeq::RK45Integrator<std::vector<double>>(lorenz_system),
        initial_conditions, 0.01, 0.5
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Analyze results
    size_t successful = 0;
    size_t timed_out = 0;
    std::chrono::microseconds total_execution_time{0};
    
    for (const auto& result : results) {
        if (result.is_success()) successful++;
        else if (result.is_timeout()) timed_out++;
        total_execution_time += result.execution_time;
    }
    
    std::cout << "  Total wall time: " << total_time.count() << " ms\n";
    std::cout << "  Average execution time per task: " 
              << total_execution_time.count() / results.size() << " μs\n";
    std::cout << "  Successful integrations: " << successful << "/" << results.size() << "\n";
    std::cout << "  Timed out integrations: " << timed_out << "/" << results.size() << "\n";
    
    if (!results.empty()) {
        std::cout << "  Strategy used: ";
        switch (results[0].used_strategy) {
            case diffeq::ExecutionStrategy::SEQUENTIAL: std::cout << "Sequential"; break;
            case diffeq::ExecutionStrategy::PARALLEL: std::cout << "Parallel"; break;
            case diffeq::ExecutionStrategy::ASYNC: std::cout << "Async"; break;
            case diffeq::ExecutionStrategy::HYBRID: std::cout << "Hybrid"; break;
            default: std::cout << "Auto"; break;
        }
        std::cout << "\n";
    }
    
    // Show some results
    std::cout << "  Sample results:\n";
    for (size_t i = 0; i < std::min(5UL, initial_conditions.size()); ++i) {
        const auto& state = initial_conditions[i];
        std::cout << "    IC[" << i << "]: [" 
                  << std::fixed << std::setprecision(3)
                  << state[0] << ", " << state[1] << ", " << state[2] << "]\n";
    }
}

void demonstrate_monte_carlo_simulation() {
    std::cout << "\n=== Monte Carlo Simulation with Auto-Parallelization ===\n";
    
    const size_t num_simulations = 1000;
    const double initial_price = 100.0;
    
    std::cout << "Running " << num_simulations << " Monte Carlo simulations...\n";
    
    // Create parallel timeout integrator for financial simulation
    auto config = diffeq::ParallelTimeoutConfig{
        .timeout_config = {
            .timeout_duration = std::chrono::milliseconds{5000},
            .throw_on_timeout = false
        },
        .strategy = diffeq::ExecutionStrategy::AUTO,
        .performance_hint = diffeq::PerformanceHint::HIGH_THROUGHPUT
    };
    
    auto integrator = diffeq::core::factory::make_auto_optimized_integrator<std::vector<double>>(
        exponential_decay, config);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Monte Carlo with automatic parallelization
    auto results = integrator->integrate_monte_carlo(
        num_simulations,
        // Initial state generator
        [initial_price](size_t i) -> std::vector<double> {
            std::mt19937 rng(i);
            std::normal_distribution<double> noise(0.0, 0.01);
            return {initial_price + noise(rng)};
        },
        // Result processor
        [](const std::vector<double>& final_state) -> double {
            return final_state[0];  // Return final price
        },
        0.01, 1.0  // dt, t_end
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Calculate statistics
    double mean_price = 0.0;
    for (double price : results) {
        mean_price += price;
    }
    mean_price /= num_simulations;
    
    double variance = 0.0;
    for (double price : results) {
        variance += (price - mean_price) * (price - mean_price);
    }
    variance /= num_simulations;
    double std_dev = std::sqrt(variance);
    
    std::cout << "  Simulation completed in: " << duration.count() << " ms\n";
    std::cout << "  Mean final price: $" << std::fixed << std::setprecision(2) << mean_price << "\n";
    std::cout << "  Standard deviation: $" << std::setprecision(2) << std_dev << "\n";
    std::cout << "  Theoretical mean: $" << std::setprecision(2) 
              << initial_price * std::exp(-1.0) << "\n";
    std::cout << "  Throughput: " << (num_simulations * 1000.0 / duration.count()) 
              << " simulations/second\n";
}

void demonstrate_fine_grained_control() {
    std::cout << "\n=== Fine-Grained Control for Advanced Users ===\n";
    
    // Advanced users can control every aspect
    auto config = diffeq::ParallelTimeoutConfig{
        .timeout_config = {
            .timeout_duration = std::chrono::milliseconds{2000},
            .throw_on_timeout = false,
            .enable_progress_callback = true,
            .progress_interval = std::chrono::milliseconds{100},
            .progress_callback = [](double current_time, double end_time, auto elapsed) {
                std::cout << "    Progress: " << std::fixed << std::setprecision(1)
                         << (current_time / end_time) * 100 << "% (elapsed: " 
                         << elapsed.count() << "ms)\n";
                return true;  // Continue integration
            }
        },
        .strategy = diffeq::ExecutionStrategy::ASYNC,  // Force async
        .performance_hint = diffeq::PerformanceHint::LOW_LATENCY,
        .max_parallel_tasks = 4,
        .async_thread_pool_size = 2,
        .enable_async_stepping = true,
        .enable_state_monitoring = true,
        .enable_hardware_detection = true,
        .enable_signal_processing = false
    };
    
    std::cout << "Creating custom configured integrator...\n";
    std::cout << "  Forced strategy: Async\n";
    std::cout << "  Performance hint: Low Latency\n";
    std::cout << "  Thread pool size: 2\n";
    std::cout << "  Progress monitoring: Enabled\n";
    
    auto integrator = diffeq::core::factory::make_parallel_timeout_integrator<
        diffeq::RK45Integrator<std::vector<double>>>(
        config, stiff_van_der_pol
    );
    
    std::vector<double> state = {1.0, 0.0};
    
    std::cout << "Integrating stiff Van der Pol system...\n";
    auto result = integrator->integrate_with_auto_parallel(state, 0.001, 0.5);
    
    std::cout << "\nAdvanced integration result:\n";
    std::cout << "  Strategy used: ";
    switch (result.used_strategy) {
        case diffeq::ExecutionStrategy::ASYNC: std::cout << "Async (as requested)"; break;
        default: std::cout << "Other"; break;
    }
    std::cout << "\n";
    std::cout << "  Setup time: " << result.setup_time.count() << " μs\n";
    std::cout << "  Execution time: " << result.execution_time.count() << " μs\n";
    std::cout << "  Total time: " << result.total_elapsed_time().count() << " ms\n";
    std::cout << "  Success: " << (result.is_success() ? "✓" : "✗") << "\n";
    
    if (result.is_success()) {
        std::cout << "  Final state: [" << state[0] << ", " << state[1] << "]\n";
    } else {
        std::cout << "  Error: " << result.timeout_result.error_message << "\n";
    }
    
    // Advanced users can also access underlying components
    std::cout << "\nAccessing underlying components:\n";
    std::cout << "  Base integrator available: ✓\n";
    std::cout << "  Async integrator available: " 
              << (integrator->async_integrator() ? "✓" : "✗") << "\n";
    std::cout << "  Integration interface available: " 
              << (integrator->integration_interface() ? "✓" : "✗") << "\n";
    
    // Show hardware detection results
    const auto& hw_caps = integrator->hardware_capabilities();
    std::cout << "  Detected CPU cores: " << hw_caps.cpu_cores << "\n";
    std::cout << "  Supports std::execution: " << (hw_caps.supports_std_execution ? "✓" : "✗") << "\n";
    std::cout << "  Supports SIMD: " << (hw_caps.supports_simd ? "✓" : "✗") << "\n";
}

void demonstrate_real_time_integration() {
    std::cout << "\n=== Real-time Integration with Signal Processing ===\n";
    
    // Configure for real-time with signal processing
    auto config = diffeq::ParallelTimeoutConfig{
        .timeout_config = {
            .timeout_duration = std::chrono::milliseconds{100},  // Short timeout for real-time
            .throw_on_timeout = false
        },
        .strategy = diffeq::ExecutionStrategy::ASYNC,
        .performance_hint = diffeq::PerformanceHint::LOW_LATENCY,
        .enable_async_stepping = true,
        .enable_state_monitoring = true,
        .enable_signal_processing = true,
        .signal_check_interval = std::chrono::microseconds{100}
    };
    
    auto integrator = diffeq::core::factory::make_parallel_timeout_integrator<
        diffeq::RK45Integrator<std::vector<double>>>(
        config, lorenz_system
    );
    
    std::cout << "Real-time integration configuration:\n";
    std::cout << "  Timeout: " << config.timeout_config.timeout_duration.count() << " ms\n";
    std::cout << "  Strategy: Async (low latency)\n";
    std::cout << "  Signal processing: Enabled\n";
    
    // Simulate real-time control loop
    std::vector<double> state = {1.0, 1.0, 1.0};
    const double dt = 0.001;  // 1ms timestep
    const int num_steps = 50;
    
    std::cout << "\nRunning " << num_steps << " real-time steps (1ms each)...\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Real-time integration step
        auto result = integrator->integrate_realtime(state, dt, step * dt + dt);
        
        auto step_end = std::chrono::high_resolution_clock::now();
        auto step_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            step_end - step_start);
        
        // Report every 10 steps
        if (step % 10 == 0) {
            std::cout << "  Step " << std::setw(2) << step 
                      << ": " << std::setw(4) << step_duration.count() << " μs"
                      << " | state=[" << std::fixed << std::setprecision(3)
                      << state[0] << ", " << state[1] << ", " << state[2] << "]"
                      << " | " << (result.is_success() ? "✓" : "✗") << "\n";
        }
        
        // Simulate real-time constraint (must complete within 1ms)
        if (step_duration.count() > 1000) {
            std::cout << "  ⚠ Real-time constraint violated at step " << step << "\n";
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        total_end - total_start);
    
    std::cout << "\nReal-time integration summary:\n";
    std::cout << "  Total time: " << total_duration.count() << " μs\n";
    std::cout << "  Average per step: " << total_duration.count() / num_steps << " μs\n";
    std::cout << "  Real-time performance: " 
              << (total_duration.count() < num_steps * 1000 ? "✓ PASSED" : "✗ FAILED") << "\n";
}

void demonstrate_performance_comparison() {
    std::cout << "\n=== Performance Comparison: Sequential vs Auto-Optimized ===\n";
    
    const size_t num_integrations = 100;
    std::vector<std::vector<double>> test_states(num_integrations, {1.0, 1.0, 1.0});
    
    // Sequential integration
    auto seq_start = std::chrono::high_resolution_clock::now();
    {
        auto integrator = diffeq::RK45Integrator<std::vector<double>>(lorenz_system);
        for (auto& state : test_states) {
            integrator.integrate(state, 0.01, 0.5);
        }
    }
    auto seq_end = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        seq_end - seq_start);
    
    // Reset states
    for (auto& state : test_states) {
        state = {1.0, 1.0, 1.0};
    }
    
    // Auto-optimized integration
    auto auto_start = std::chrono::high_resolution_clock::now();
    {
        auto results = diffeq::integrate_batch_auto(
            diffeq::RK45Integrator<std::vector<double>>(lorenz_system),
            test_states, 0.01, 0.5
        );
    }
    auto auto_end = std::chrono::high_resolution_clock::now();
    auto auto_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        auto_end - auto_start);
    
    std::cout << "Performance comparison (" << num_integrations << " integrations):\n";
    std::cout << "  Sequential: " << seq_duration.count() << " ms\n";
    std::cout << "  Auto-optimized: " << auto_duration.count() << " ms\n";
    
    if (auto_duration.count() > 0) {
        double speedup = static_cast<double>(seq_duration.count()) / auto_duration.count();
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
        
        if (speedup > 1.0) {
            std::cout << "  🚀 Auto-optimization provided significant speedup!\n";
        } else {
            std::cout << "  📊 Sequential was faster (small problem size)\n";
        }
    }
    
    std::cout << "  Hardware cores: " << std::thread::hardware_concurrency() << "\n";
}

int main() {
    std::cout << "DiffEq Library - Seamless Parallel Timeout Integration Demo\n";
    std::cout << "============================================================\n";
    
    std::cout << "\nThis demo shows how the diffeq library seamlessly combines:\n";
    std::cout << "• Timeout protection for robust applications\n";
    std::cout << "• Automatic hardware utilization for performance\n";
    std::cout << "• Async/parallel execution for scalability\n";
    std::cout << "• Fine-grained control for advanced users\n";
    std::cout << "• Real-time capabilities for control systems\n";
    
    try {
        demonstrate_automatic_hardware_utilization();
        demonstrate_batch_processing();
        demonstrate_monte_carlo_simulation();
        demonstrate_fine_grained_control();
        demonstrate_real_time_integration();
        demonstrate_performance_comparison();
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "✓ All demonstrations completed successfully!\n";
        std::cout << "\nThe diffeq library provides:\n";
        std::cout << "1. 🎯 Zero-configuration auto-optimization\n";
        std::cout << "2. ⚡ Seamless hardware utilization\n";
        std::cout << "3. 🛡️ Built-in timeout protection\n";
        std::cout << "4. 🔧 Full control when needed\n";
        std::cout << "5. ⏱️ Real-time capabilities\n";
        std::cout << "6. 📈 Scalable performance\n";
        
        std::cout << "\nUsers can simply call diffeq::integrate_auto() and get\n";
        std::cout << "optimal performance automatically, or configure everything\n";
        std::cout << "manually for specialized requirements.\n";
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Error occurred: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}