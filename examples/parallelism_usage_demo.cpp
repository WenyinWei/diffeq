#include <diffeq.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>

/**
 * @brief Quick Start Example - Simplified Parallel Interface
 * 
 * This shows the easiest way to add parallelism to your diffeq computations.
 * No complex configuration needed!
 */
void quick_start_example() {
    std::cout << "\n=== Quick Start: Simplified Parallel Interface ===" << std::endl;
    
    // Example: Parallel ODE integration for multiple initial conditions
    std::vector<std::vector<double>> initial_conditions;
    for (int i = 0; i < 100; ++i) {
        initial_conditions.push_back({static_cast<double>(i), 0.0});
    }
    
    // Simple exponential decay: dy/dt = -0.1 * y
    auto system = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = -0.1 * y[0];
        dydt[1] = -0.2 * y[1];
    };
    
    std::cout << "Integrating " << initial_conditions.size() << " initial conditions in parallel..." << std::endl;
    
    // THIS IS ALL YOU NEED FOR PARALLEL EXECUTION!
    diffeq::execution::parallel_for_each(initial_conditions, [&](std::vector<double>& state) {
        diffeq::RK4Integrator<std::vector<double>> integrator(system);
        integrator.step(state, 0.01); // Single integration step
    });
    
    std::cout << "✓ Parallel integration completed!" << std::endl;
    std::cout << "Result for initial condition 10: [" << initial_conditions[10][0] 
              << ", " << initial_conditions[10][1] << "]" << std::endl;
    
    // Want to use GPU if available? Just one line:
    diffeq::execution::enable_gpu_acceleration();
    
    // Want more workers? Just one line:
    diffeq::execution::set_parallel_workers(8);
    
    std::cout << "Current worker count: " << diffeq::execution::parallel().worker_count() << std::endl;
}

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
    std::cout << "\n=== Robotics Control System with Real-time Parallelism ===" << std::endl;
    
    // SIMPLE APPROACH: Use the simplified parallel interface for basic needs
    std::cout << "\n--- Simple Parallel Approach ---" << std::endl;
    
    // Setup multiple control systems (e.g., different robot joints)
    std::vector<std::vector<double>> joint_states;
    for (int i = 0; i < 6; ++i) {  // 6-DOF robot arm
        joint_states.push_back({0.1 * i, 0.0}); // [angle, angular_velocity]
    }
    
    // Create simple parallel executor
    auto parallel = diffeq::execution::Parallel(4); // 4 worker threads
    
    auto simple_start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel control loop - very simple!
    parallel.for_each(joint_states, [](std::vector<double>& state) {
        RobotArmSystem system;
        diffeq::RK4Integrator<std::vector<double>> integrator(system);
        integrator.step(state, 0.001); // 1ms control timestep
    });
    
    auto simple_end_time = std::chrono::high_resolution_clock::now();
    auto simple_duration = std::chrono::duration_cast<std::chrono::microseconds>(simple_end_time - simple_start_time);
    
    std::cout << "Simple parallel control completed in " << simple_duration.count() << " μs" << std::endl;
    std::cout << "Average per joint: " << simple_duration.count() / joint_states.size() << " μs" << std::endl;
    
    // ADVANCED APPROACH: Use full facade for complex real-time requirements
    std::cout << "\n--- Advanced Facade Approach (for complex scenarios) ---" << std::endl;
    
    // For applications requiring precise real-time control, load balancing,
    // hardware-specific optimizations, etc., use the full facade:
    auto parallel_config = diffeq::execution::presets::robotics_control()
        .realtime_priority()
        .workers(4)                    // Dedicated cores for control
        .batch_size(1)                 // Single step processing
        .disable_load_balancing()      // Deterministic scheduling
        .build();
    
    // Create integrator for robot dynamics
    auto robot_system = RobotArmSystem{};
    auto integrator = diffeq::RK4Integrator<std::vector<double>>(robot_system);
    
    // Initial state: [angle=0.1 rad, angular_velocity=0]
    std::vector<double> state = {0.1, 0.0};
    double dt = 0.001;  // 1ms timestep for 1kHz control
    double simulation_time = 1.0;
    
    std::cout << "Running real-time robot control simulation..." << std::endl;
    std::cout << "Target frequency: 1kHz (1ms timestep)" << std::endl;
    
    auto advanced_start_time = std::chrono::high_resolution_clock::now();
    
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
                      << " rad, measured=" << measured_angle << " rad" << std::endl;
        }
    }
    
    auto advanced_end_time = std::chrono::high_resolution_clock::now();
    auto advanced_duration = std::chrono::duration_cast<std::chrono::milliseconds>(advanced_end_time - advanced_start_time);
    
    std::cout << "Simulation completed in " << advanced_duration.count() << "ms" << std::endl;
    std::cout << "Average loop time: " << advanced_duration.count() / (simulation_time / dt) << "ms" << std::endl;
    std::cout << "Final robot state: angle=" << state[0] << " rad, velocity=" << state[1] << " rad/s" << std::endl;
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
    std::cout << "\n=== Stochastic Process Research with GPU-Accelerated Monte Carlo ===" << std::endl;
    
    // Configure for high-throughput Monte Carlo
    auto parallel_config = diffeq::execution::presets::monte_carlo()
        .workers(16)                   // Many workers for throughput
        .batch_size(1000)              // Large batches for efficiency
        .enable_gpu_if_available()     // Use GPU for massive parallelism
        .build();
    
    const int num_simulations = 10000;
    const double initial_price = 100.0;
    const double time_horizon = 1.0;  // 1 year
    const double dt = 0.01;           // Daily steps
    
    std::cout << "Running " << num_simulations << " Monte Carlo simulations..." << std::endl;
    std::cout << "Using " << parallel_config->worker_count() << " workers" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel Monte Carlo simulation
    std::vector<double> final_prices(num_simulations);
    
    parallel_config->parallel_for(0, num_simulations, [&](int i) {
        // Each simulation gets its own random number generator
        std::mt19937 rng(i);  // Seed with simulation index
        std::normal_distribution<double> normal(0.0, 1.0);
        
        GeometricBrownianMotion gbm;
        diffeq::EulerMaruyama<std::vector<double>> integrator(gbm);
        
        std::vector<double> state = {initial_price};
        
        // Integrate stochastic process
        for (double t = 0.0; t < time_horizon; t += dt) {
            integrator.step(state, dt, normal(rng));
        }
        
        final_prices[i] = state[0];
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate statistics
    double sum = 0.0, sum_sq = 0.0;
    for (double price : final_prices) {
        sum += price;
        sum_sq += price * price;
    }
    double mean = sum / num_simulations;
    double variance = (sum_sq / num_simulations) - (mean * mean);
    double std_dev = std::sqrt(variance);
    
    std::cout << "Monte Carlo completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Simulations per second: " << (num_simulations * 1000.0 / duration.count()) << std::endl;
    std::cout << "Final price statistics:" << std::endl;
    std::cout << "  Mean: $" << mean << std::endl;
    std::cout << "  Std Dev: $" << std_dev << std::endl;
    std::cout << "  Min: $" << *std::min_element(final_prices.begin(), final_prices.end()) << std::endl;
    std::cout << "  Max: $" << *std::max_element(final_prices.begin(), final_prices.end()) << std::endl;
}

} // namespace stochastic_research

/**
 * @brief Multi-Hardware Target Example
 * 
 * Demonstrates how to benchmark and choose the best hardware target
 * for different problem sizes and requirements.
 */
namespace multi_hardware_demo {

struct ExponentialDecay {
    double k = 1.0;
    
    void operator()(double t, const std::vector<double>& state, std::vector<double>& derivative) {
        derivative[0] = -k * state[0];
    }
};

void benchmark_hardware_targets() {
    std::cout << "\n=== Multi-Hardware Target Benchmarking ===" << std::endl;
    
    const int num_integrations = 10000;
    std::vector<double> initial_state = {1.0};
    
    ExponentialDecay system;
    diffeq::RK4Integrator<std::vector<double>> integrator(system);
    
    std::cout << "Benchmarking " << num_integrations << " integrations on different targets..." << std::endl;
    
    // Test CPU-only execution
    auto cpu_config = diffeq::execution::presets::cpu_only()
        .workers(4)
        .build();
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<double>> states(num_integrations, initial_state);
    cpu_config->parallel_for(0, num_integrations, [&](int i) {
        integrator.step(states[i], 0.01);
    });
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU (4 cores): " << cpu_duration.count() << " μs" << std::endl;
    
    // Test GPU execution (if available)
    if (diffeq::execution::gpu_available()) {
        auto gpu_config = diffeq::execution::presets::gpu_accelerated()
            .build();
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<double>> gpu_states(num_integrations, initial_state);
        gpu_config->parallel_for(0, num_integrations, [&](int i) {
            integrator.step(gpu_states[i], 0.01);
        });
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
        
        std::cout << "GPU: " << gpu_duration.count() << " μs" << std::endl;
        std::cout << "GPU speedup: " << (double)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    } else {
        std::cout << "GPU: Not available" << std::endl;
    }
    
    // Test hybrid CPU+GPU execution
    auto hybrid_config = diffeq::execution::presets::hybrid()
        .cpu_workers(2)
        .gpu_batch_size(1000)
        .build();
    
    auto hybrid_start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<double>> hybrid_states(num_integrations, initial_state);
    hybrid_config->parallel_for(0, num_integrations, [&](int i) {
        integrator.step(hybrid_states[i], 0.01);
    });
    
    auto hybrid_end = std::chrono::high_resolution_clock::now();
    auto hybrid_duration = std::chrono::duration_cast<std::chrono::microseconds>(hybrid_end - hybrid_start);
    
    std::cout << "Hybrid (CPU+GPU): " << hybrid_duration.count() << " μs" << std::endl;
}

} // namespace multi_hardware_demo

/**
 * @brief Comprehensive demonstration of all parallelism features
 */
void demonstrate_all_parallelism_features() {
    std::cout << "\n=== Comprehensive Parallelism Feature Demo ===" << std::endl;
    
    // 1. Quick start
    quick_start_example();
    
    // 2. Robotics control
    robotics_control::demonstrate_realtime_control();
    
    // 3. Stochastic research
    stochastic_research::demonstrate_monte_carlo_simulation();
    
    // 4. Hardware benchmarking
    multi_hardware_demo::benchmark_hardware_targets();
    
    std::cout << "\n=== All parallelism features demonstrated! ===" << std::endl;
}

int main() {
    std::cout << "=== diffeq Parallelism Usage Examples ===" << std::endl;
    
    // Run comprehensive demonstration
    demonstrate_all_parallelism_features();
    
    return 0;
} 