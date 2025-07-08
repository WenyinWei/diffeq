#include <diffeq.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <thread>
#include <future>
#include <algorithm>
#include <execution>
#include <ranges>
#include <numeric>

/**
 * @brief Modern Parallel Integration Example
 * 
 * This demonstrates how to use standard C++20/23 parallelism with diffeq
 * integrators for high-performance computations.
 */
void modern_parallel_integration_example() {
    std::cout << "\n=== Modern Parallel Integration Example ===" << std::endl;
    
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
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use standard C++20 parallel execution
    std::for_each(std::execution::par, initial_conditions.begin(), initial_conditions.end(),
        [&](std::vector<double>& state) {
            auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(system);
            integrator->step(state, 0.01); // Single integration step
        });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "[PASS] Parallel integration completed in " << duration.count() << " us!" << std::endl;
    std::cout << "Result for initial condition 10: [" << initial_conditions[10][0] 
              << ", " << initial_conditions[10][1] << "]" << std::endl;
}

/**
 * @brief Robotics Control Systems Example
 * 
 * Demonstrates real-time integration with feedback signals from sensors
 * using modern async capabilities.
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
    std::cout << "\n=== Robotics Control System with Modern Async ===" << std::endl;
    
    // Setup multiple control systems (e.g., different robot joints)
    std::vector<std::vector<double>> joint_states;
    for (int i = 0; i < 6; ++i) {  // 6-DOF robot arm
        joint_states.push_back({0.1 * i, 0.0}); // [angle, angular_velocity]
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel control loop using standard C++ parallelism
    std::for_each(std::execution::par, joint_states.begin(), joint_states.end(),
        [](std::vector<double>& state) {
            RobotArmSystem system;
            auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(system);
            integrator->step(state, 0.001); // 1ms control timestep
        });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Parallel control completed in " << duration.count() << " us" << std::endl;
    std::cout << "Average per joint: " << duration.count() / joint_states.size() << " us" << std::endl;
    
    // Advanced async approach using diffeq async capabilities
    std::cout << "\n--- Advanced Async Approach ---" << std::endl;
    
    // Create integrator for robot dynamics
    auto robot_system = RobotArmSystem{};
    auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(robot_system);
    
    // Initial state: [angle=0.1 rad, angular_velocity=0]
    std::vector<double> state = {0.1, 0.0};
    double dt = 0.001;  // 1ms timestep for 1kHz control
    double simulation_time = 1.0;
    
    std::cout << "Running real-time robot control simulation..." << std::endl;
    std::cout << "Target frequency: 1kHz (1ms timestep)" << std::endl;
    
    auto advanced_start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate real-time control loop with async execution
    std::vector<std::future<void>> control_futures;
    
    for (double t = 0.0; t < simulation_time; t += dt) {
        // Execute control step asynchronously
        auto control_future = std::async(std::launch::async, [&, t]() {
            integrator->step(state, dt);
        });
        
        // Simulate sensor reading (parallel)
        auto sensor_future = std::async(std::launch::async, [&]() {
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
 * Demonstrates Monte Carlo simulations using modern parallelism.
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
    std::cout << "\n=== Stochastic Process Research with Modern Parallelism ===" << std::endl;
    
    const int num_simulations = 10000;
    const double initial_price = 100.0;
    const double dt = 0.01;
    const double t_final = 1.0;
    
    std::cout << "Running " << num_simulations << " Monte Carlo simulations..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use standard C++20 parallel execution for Monte Carlo
    std::vector<double> final_prices(num_simulations);
    
    std::for_each(std::execution::par, 
        std::views::iota(0, num_simulations).begin(),
        std::views::iota(0, num_simulations).end(),
        [&](int i) {
            std::mt19937 rng(i);  // Seed with simulation index
            std::normal_distribution<double> normal(0.0, 1.0);
            
            // Define SDE parameters
            const double mu = 0.05;    // drift coefficient
            const double sigma = 0.2;  // volatility coefficient
            
            // Simple Euler-Maruyama implementation
            std::vector<double> state = {initial_price};
            for (double t = 0.0; t < t_final; t += dt) {
                double dW = normal(rng) * std::sqrt(dt);
                state[0] += mu * state[0] * dt + sigma * state[0] * dW;
            }
            
            final_prices[i] = state[0];
        });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Calculate statistics
    double mean_price = std::reduce(final_prices.begin(), final_prices.end()) / num_simulations;
    double variance = 0.0;
    for (double price : final_prices) {
        variance += (price - mean_price) * (price - mean_price);
    }
    variance /= num_simulations;
    double std_dev = std::sqrt(variance);
    
    std::cout << "Monte Carlo simulation completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Mean final price: " << mean_price << std::endl;
    std::cout << "Standard deviation: " << std_dev << std::endl;
    std::cout << "Theoretical mean: " << initial_price * std::exp(0.05) << std::endl;
}

} // namespace stochastic_research

/**
 * @brief Hardware Benchmarking Example
 * 
 * Demonstrates performance comparison across different hardware configurations.
 */
namespace hardware_benchmark {

struct ExponentialDecay {
    double k = 1.0;
    
    void operator()(double t, const std::vector<double>& state, std::vector<double>& derivative) {
        derivative[0] = -k * state[0];
    }
};

void benchmark_hardware_targets() {
    std::cout << "\n=== Hardware Performance Benchmarking ===" << std::endl;
    
    const int num_integrations = 1000;  // Reduced from 10000 to 1000 for faster execution
    const double dt = 0.01;
    const double t_final = 1.0;
    
    auto system = ExponentialDecay{};
    auto integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(system);
    std::vector<double> initial_state = {1.0};
    
    // Sequential execution
    auto seq_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> seq_states(num_integrations, initial_state);
            for (int i = 0; i < num_integrations; ++i) {
            integrator->integrate(seq_states[i], dt, t_final);
        }
    auto seq_end = std::chrono::high_resolution_clock::now();
    auto seq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start);
    
    // Parallel execution
    auto par_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> par_states(num_integrations, initial_state);
    std::for_each(std::execution::par, par_states.begin(), par_states.end(),
        [&](std::vector<double>& state) {
            auto local_integrator = std::make_unique<diffeq::RK45Integrator<std::vector<double>>>(system);
            local_integrator->integrate(state, dt, t_final);
        });
    auto par_end = std::chrono::high_resolution_clock::now();
    auto par_duration = std::chrono::duration_cast<std::chrono::milliseconds>(par_end - par_start);
    
    std::cout << "Sequential execution: " << seq_duration.count() << "ms" << std::endl;
    std::cout << "Parallel execution: " << par_duration.count() << "ms" << std::endl;
    std::cout << "Speedup: " << static_cast<double>(seq_duration.count()) / par_duration.count() << "x" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
}

} // namespace hardware_benchmark

void demonstrate_all_parallelism_features() {
    std::cout << "\n=== Complete Parallelism Feature Demonstration ===" << std::endl;
    
    modern_parallel_integration_example();
    robotics_control::demonstrate_realtime_control();
    stochastic_research::demonstrate_monte_carlo_simulation();
    hardware_benchmark::benchmark_hardware_targets();
    
    std::cout << "\n=== All Examples Completed Successfully! ===" << std::endl;
}

int main() {
    std::cout << "Modern DiffeQ Parallelism Examples" << std::endl;
    std::cout << "===================================" << std::endl;
    
    demonstrate_all_parallelism_features();
    
    return 0;
} 