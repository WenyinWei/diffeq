#pragma once

#include <diffeq.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

namespace diffeq::examples::parallelism {

/**
 * @brief Robotics Control Systems Example
 * 
 * Demonstrates real-time integration with feedback signals from sensors
 * with low latency and deterministic performance requirements.
 */
namespace robotics_control {

// Robot arm dynamics (simplified)
struct RobotArmSystem {
    double mass = 1.0;
    double length = 0.5;
    double damping = 0.1;
    double gravity = 9.81;
    
    void operator()(double t, const std::vector<double>& state, std::vector<double>& derivative) {
        // state = [angle, angular_velocity]
        // Simple pendulum with control input
        double angle = state[0];
        double angular_vel = state[1];
        
        // Control input (placeholder - would come from feedback controller)
        double control_torque = -2.0 * angle - 0.5 * angular_vel;  // PD controller
        
        derivative[0] = angular_vel;
        derivative[1] = -(gravity / length) * std::sin(angle) - 
                        (damping / (mass * length * length)) * angular_vel + 
                        control_torque / (mass * length * length);
    }
};

void demonstrate_realtime_control() {
    std::cout << "\n=== Robotics Control System with Real-time Parallelism ===\n";
    
    // Configure for real-time robotics control
    auto parallel_config = diffeq::execution::presets::robotics_control()
        .realtime_priority()
        .workers(4)                    // Dedicated cores for control
        .batch_size(1)                 // Single step processing
        .disable_load_balancing()      // Deterministic scheduling
        .build();
    
    // Create integrator for robot dynamics
    auto robot_system = RobotArmSystem{};
    auto integrator = diffeq::ode::factory::make_rk4_integrator<std::vector<double>, double>(robot_system);
    
    // Initial state: [angle=0.1 rad, angular_velocity=0]
    std::vector<double> state = {0.1, 0.0};
    double dt = 0.001;  // 1ms timestep for 1kHz control
    double simulation_time = 1.0;
    
    std::cout << "Running real-time robot control simulation...\n";
    std::cout << "Target frequency: 1kHz (1ms timestep)\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate real-time control loop
    for (double t = 0.0; t < simulation_time; t += dt) {
        // Execute control step with real-time priority
        auto control_future = parallel_config->async([&]() {
            integrator->step(state, dt);
        });
        
        // Simulate sensor reading (parallel)
        auto sensor_future = parallel_config->async([&]() {
            // Placeholder for sensor data processing
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            return state[0];  // Return angle measurement
        });
        
        // Wait for both to complete
        control_future.wait();
        double measured_angle = sensor_future.get();
        
        // Output every 100ms
        if (static_cast<int>(t * 1000) % 100 == 0) {
            std::cout << "t=" << t << "s, angle=" << state[0] 
                      << " rad, measured=" << measured_angle << " rad\n";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Simulation completed in " << duration.count() << "ms\n";
    std::cout << "Average loop time: " << duration.count() / (simulation_time / dt) << "ms\n";
    std::cout << "Final robot state: angle=" << state[0] << " rad, velocity=" << state[1] << " rad/s\n";
}

} // namespace robotics_control

/**
 * @brief Stochastic Process Research Example
 * 
 * Demonstrates Monte Carlo simulations requiring thousands/millions of integrations
 * with focus on throughput rather than latency.
 */
namespace stochastic_research {

// Geometric Brownian Motion for financial modeling
struct GeometricBrownianMotion {
    double mu = 0.05;    // drift
    double sigma = 0.2;  // volatility
    
    void operator()(double t, const std::vector<double>& state, std::vector<double>& derivative) {
        derivative[0] = mu * state[0];  // deterministic part
    }
    
    void diffusion(double t, const std::vector<double>& state, std::vector<double>& noise_coeff) {
        noise_coeff[0] = sigma * state[0];  // stochastic part
    }
};

void demonstrate_monte_carlo_simulation() {
    std::cout << "\n=== Stochastic Process Research with GPU-Accelerated Monte Carlo ===\n";
    
    // Configure for high-throughput Monte Carlo
    auto parallel_config = diffeq::execution::presets::monte_carlo()
        .auto_target()                 // Let system choose best hardware
        .workers(std::thread::hardware_concurrency() * 4)  // Oversubscribe for throughput
        .batch_size(10000)            // Large batches for efficiency
        .enable_load_balancing()      // Dynamic load balancing
        .build();
    
    // Print available hardware
    diffeq::execution::hardware::HardwareCapabilities::print_capabilities();
    
    const size_t num_simulations = 100000;
    const double T = 1.0;           // 1 year
    const double dt = 1.0/252;      // daily steps
    const double S0 = 100.0;        // initial stock price
    
    std::cout << "\nRunning " << num_simulations << " Monte Carlo simulations...\n";
    std::cout << "Each simulation: " << static_cast<int>(T/dt) << " steps over " << T << " years\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare simulation parameters
    std::vector<double> final_prices(num_simulations);
    std::vector<std::future<double>> simulation_futures;
    
    // Create the SDE system
    auto gbm_system = GeometricBrownianMotion{};
    
    // Launch parallel simulations
    for (size_t i = 0; i < num_simulations; ++i) {
        simulation_futures.push_back(parallel_config->async([=]() {
            // Create integrator for this simulation
            auto integrator = diffeq::ode::factory::make_rk4_integrator<std::vector<double>, double>(gbm_system);
            
            // Random number generator for this thread
            std::mt19937 rng(std::random_device{}() + i);
            std::normal_distribution<double> normal(0.0, 1.0);
            
            std::vector<double> state = {S0};
            double t = 0.0;
            
            while (t < T) {
                // Deterministic step
                integrator->step(state, dt);
                
                // Add stochastic component (simplified Euler-Maruyama)
                double dW = normal(rng) * std::sqrt(dt);
                state[0] += gbm_system.sigma * state[0] * dW;
                
                t += dt;
            }
            
            return state[0];  // Return final price
        }));
    }
    
    // Collect results
    for (size_t i = 0; i < num_simulations; ++i) {
        final_prices[i] = simulation_futures[i].get();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate statistics
    double mean_price = 0.0;
    double min_price = final_prices[0];
    double max_price = final_prices[0];
    
    for (double price : final_prices) {
        mean_price += price;
        min_price = std::min(min_price, price);
        max_price = std::max(max_price, price);
    }
    mean_price /= num_simulations;
    
    // Calculate variance
    double variance = 0.0;
    for (double price : final_prices) {
        variance += (price - mean_price) * (price - mean_price);
    }
    variance /= (num_simulations - 1);
    
    std::cout << "\nMonte Carlo Results:\n";
    std::cout << "  Simulation time: " << duration.count() << "ms\n";
    std::cout << "  Throughput: " << (num_simulations * 1000.0) / duration.count() << " simulations/second\n";
    std::cout << "  Mean final price: $" << mean_price << "\n";
    std::cout << "  Price range: $" << min_price << " - $" << max_price << "\n";
    std::cout << "  Standard deviation: $" << std::sqrt(variance) << "\n";
    std::cout << "  Expected price (theoretical): $" << S0 * std::exp(gbm_system.mu * T) << "\n";
}

} // namespace stochastic_research

/**
 * @brief Multi-Hardware Benchmark
 * 
 * Demonstrates the same computation running on different hardware targets
 * to showcase the unified interface abstraction.
 */
namespace multi_hardware_demo {

// Simple ODE for benchmarking: dx/dt = -k*x (exponential decay)
struct ExponentialDecay {
    double k = 1.0;
    
    void operator()(double t, const std::vector<double>& state, std::vector<double>& derivative) {
        derivative[0] = -k * state[0];
    }
};

void benchmark_hardware_targets() {
    std::cout << "\n=== Multi-Hardware Parallelism Benchmark ===\n";
    
    const size_t num_integrations = 10000;
    const double dt = 0.01;
    const double end_time = 1.0;
    const std::vector<double> initial_state = {1.0};
    
    std::vector<diffeq::execution::HardwareTarget> targets = {
        diffeq::execution::HardwareTarget::CPU_Sequential,
        diffeq::execution::HardwareTarget::CPU_ThreadPool,
        diffeq::execution::HardwareTarget::GPU_CUDA,
        diffeq::execution::HardwareTarget::GPU_OpenCL,
        diffeq::execution::HardwareTarget::FPGA_HLS
    };
    
    auto system = ExponentialDecay{};
    
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
        
        std::cout << "\nBenchmarking " << target_name << "...\n";
        
        // Configure parallelism for this hardware target
        diffeq::execution::ParallelConfig config;
        config.target = target;
        config.max_workers = std::thread::hardware_concurrency();
        config.batch_size = num_integrations / 10;
        
        auto parallel_facade = std::make_unique<diffeq::execution::ParallelismFacade>(config);
        
        if (!parallel_facade->is_target_available(target)) {
            std::cout << "  Hardware not available, skipping...\n";
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create vector of initial conditions
        std::vector<std::vector<double>> states(num_integrations, initial_state);
        
        // Parallel integration using the unified interface
        parallel_facade->parallel_for_each(states.begin(), states.end(), [&](auto& state) {
            auto integrator = diffeq::ode::factory::make_rk4_integrator<std::vector<double>, double>(system);
            
            double t = 0.0;
            while (t < end_time) {
                double step_size = std::min(dt, end_time - t);
                integrator->step(state, step_size);
                t += step_size;
            }
        });
        
        auto end_time_chrono = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_chrono - start_time);
        
        // Verify results (should be approximately e^(-k*t) = e^(-1) ≈ 0.368)
        double expected_result = std::exp(-system.k * end_time);
        double mean_result = 0.0;
        for (const auto& state : states) {
            mean_result += state[0];
        }
        mean_result /= num_integrations;
        
        double error = std::abs(mean_result - expected_result);
        
        std::cout << "  Execution time: " << duration.count() << "ms\n";
        std::cout << "  Throughput: " << (num_integrations * 1000.0) / duration.count() << " integrations/second\n";
        std::cout << "  Average result: " << mean_result << " (expected: " << expected_result << ")\n";
        std::cout << "  Error: " << error << "\n";
        std::cout << "  Max concurrency: " << parallel_facade->get_max_concurrency() << "\n";
    }
}

} // namespace multi_hardware_demo

/**
 * @brief Demonstration function that runs all examples
 */
void demonstrate_all_parallelism_features() {
    std::cout << "=== Enhanced Parallelism Capabilities Demo ===\n";
    std::cout << "Demonstrating modern C++ parallelism with unified hardware interface\n";
    
    // Run robotics control example
    robotics_control::demonstrate_realtime_control();
    
    // Run stochastic research example
    stochastic_research::demonstrate_monte_carlo_simulation();
    
    // Run multi-hardware benchmark
    multi_hardware_demo::benchmark_hardware_targets();
    
    std::cout << "\n=== All Parallelism Demonstrations Complete ===\n";
}

} // namespace diffeq::examples::parallelism