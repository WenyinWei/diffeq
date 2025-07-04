#include <execution/parallelism_facade_clean.hpp>
#include <execution/parallel_builder.hpp>
#include <integrators/ode/rk4.hpp>
#include <integrators/ode/euler.hpp>
#include <integrators/ode/rk45.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

/**
 * @brief Integration Example with Enhanced Parallelism
 * 
 * This example demonstrates how the new parallelism capabilities 
 * integrate seamlessly with existing diffeq integrators.
 */

int main() {
    std::cout << "=== Enhanced Parallelism Integration with ODE Solvers ===\n\n";
    
    // Example 1: Parallel integration of multiple initial conditions
    std::cout << "1. Parallel Integration of Multiple Initial Conditions\n";
    std::cout << "   Solving harmonic oscillator: d²x/dt² = -ω²x with different initial conditions\n\n";
    
    // Define the harmonic oscillator system
    auto harmonic_oscillator = [](double /*t*/, const std::vector<double>& state, std::vector<double>& derivative) {
        const double omega = 1.0;  // Natural frequency
        derivative[0] = state[1];                    // dx/dt = v
        derivative[1] = -omega * omega * state[0];   // dv/dt = -ω²x
    };
    
    // Create multiple initial conditions
    std::vector<std::vector<double>> initial_conditions = {
        {1.0, 0.0},   // x=1, v=0
        {0.0, 1.0},   // x=0, v=1
        {0.5, 0.5},   // x=0.5, v=0.5
        {-1.0, 0.0},  // x=-1, v=0
        {0.0, -1.0},  // x=0, v=-1
        {2.0, 0.0},   // x=2, v=0
        {0.0, 2.0},   // x=0, v=2
        {1.5, -0.5}   // x=1.5, v=-0.5
    };
    
    // Configure parallelism for optimal performance
    auto parallel_facade = diffeq::execution::parallel_execution()
        .target_cpu()
        .use_thread_pool()
        .workers(std::thread::hardware_concurrency())
        .normal_priority()
        .enable_load_balancing()
        .build();
    
    std::cout << "Using " << parallel_facade->get_max_concurrency() << " worker threads\n";
    
    const double dt = 0.01;
    const double end_time = 2.0 * M_PI;  // One full period
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Parallel integration using the facade
    parallel_facade->parallel_for_each(initial_conditions.begin(), initial_conditions.end(), 
                                      [&](std::vector<double>& state) {
        // Create integrator for this thread (RK4 for accuracy)
        auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(harmonic_oscillator);
        
        double t = 0.0;
        while (t < end_time) {
            double step_size = std::min(dt, end_time - t);
            integrator.step(state, step_size);
            t += step_size;
        }
    });
    
    auto end_time_chrono = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_chrono - start_time);
    
    std::cout << "Integration completed in " << duration.count() << "ms\n";
    std::cout << "Final states (should be close to initial due to periodicity):\n";
    for (size_t i = 0; i < initial_conditions.size(); ++i) {
        const auto& state = initial_conditions[i];
        std::cout << "  Condition " << i+1 << ": x=" << state[0] << ", v=" << state[1] << "\n";
    }
    
    // Example 2: Monte Carlo simulation with stochastic differential equations
    std::cout << "\n2. Monte Carlo SDE Simulation with GPU-Style Parallelism\n";
    std::cout << "   Simulating geometric Brownian motion for financial modeling\n\n";
    
    // Configure for high-throughput Monte Carlo
    auto monte_carlo_facade = diffeq::execution::presets::monte_carlo().build();
    
    const size_t num_simulations = 1000;  // Reduced for demo
    const double S0 = 100.0;  // Initial stock price
    const double mu = 0.05;   // Drift
    const double sigma = 0.2; // Volatility
    const double T = 1.0;     // Time horizon
    
    std::cout << "Running " << num_simulations << " Monte Carlo simulations...\n";
    
    // Create storage for results
    std::vector<double> final_prices(num_simulations);
    std::vector<std::future<double>> simulation_futures;
    
    start_time = std::chrono::high_resolution_clock::now();
    
    // Launch parallel simulations
    for (size_t i = 0; i < num_simulations; ++i) {
        simulation_futures.push_back(monte_carlo_facade->async([=]() {
            // Each simulation uses its own random number generator
            std::mt19937 rng(std::random_device{}() + i);
            std::normal_distribution<double> normal(0.0, 1.0);
            
            // Simple Geometric Brownian Motion system
            auto gbm_system = [mu](double /*t*/, const std::vector<double>& state, std::vector<double>& derivative) {
                derivative[0] = mu * state[0];  // Drift term
            };
            
            auto integrator = diffeq::integrators::ode::EulerIntegrator<std::vector<double>, double>(gbm_system);
            
            std::vector<double> state = {S0};
            double t = 0.0;
            const double dt_sim = 0.01;
            
            while (t < T) {
                // Deterministic step
                integrator.step(state, dt_sim);
                
                // Add stochastic component (simplified Euler-Maruyama)
                double dW = normal(rng) * std::sqrt(dt_sim);
                state[0] += sigma * state[0] * dW;
                
                t += dt_sim;
            }
            
            return state[0];  // Return final price
        }));
    }
    
    // Collect results
    for (size_t i = 0; i < num_simulations; ++i) {
        final_prices[i] = simulation_futures[i].get();
    }
    
    end_time_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_chrono - start_time);
    
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
    
    double variance = 0.0;
    for (double price : final_prices) {
        variance += (price - mean_price) * (price - mean_price);
    }
    variance /= (num_simulations - 1);
    
    std::cout << "Monte Carlo Results:\n";
    std::cout << "  Simulation time: " << duration.count() << "ms\n";
    std::cout << "  Throughput: " << (num_simulations * 1000.0) / duration.count() << " simulations/second\n";
    std::cout << "  Mean final price: $" << mean_price << "\n";
    std::cout << "  Price range: $" << min_price << " - $" << max_price << "\n";
    std::cout << "  Standard deviation: $" << std::sqrt(variance) << "\n";
    std::cout << "  Expected price (theoretical): $" << S0 * std::exp(mu * T) << "\n";
    
    // Example 3: Real-time integration for control systems
    std::cout << "\n3. Real-time Integration for Control Systems\n";
    std::cout << "   Simulating a controlled pendulum with real-time constraints\n\n";
    
    // Configure for real-time execution
    auto realtime_facade = diffeq::execution::presets::real_time_systems().build();
    
    // Controlled pendulum system
    auto controlled_pendulum = [](double t, const std::vector<double>& state, std::vector<double>& derivative) {
        const double g = 9.81;  // Gravity
        const double L = 1.0;   // Pendulum length
        const double b = 0.1;   // Damping
        
        double theta = state[0];      // Angle
        double theta_dot = state[1];  // Angular velocity
        
        // Simple PD controller
        double u = -10.0 * theta - 2.0 * theta_dot;  // Control torque
        
        derivative[0] = theta_dot;
        derivative[1] = -(g/L) * std::sin(theta) - b * theta_dot + u;
    };
    
    std::vector<double> pendulum_state = {0.5, 0.0};  // Initial angle 0.5 rad, no initial velocity
    const double control_dt = 0.001;  // 1ms control loop (1kHz)
    const double control_time = 1.0;   // 1 second of control (reduced for demo)
    
    std::cout << "Running real-time control loop at 1kHz for " << control_time << " seconds...\n";
    
    start_time = std::chrono::high_resolution_clock::now();
    auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>, double>(controlled_pendulum);
    
    size_t steps = 0;
    for (double t = 0.0; t < control_time; t += control_dt) {
        // Execute one control step with real-time priority
        auto control_future = realtime_facade->async([&]() {
            integrator.step(pendulum_state, control_dt);
        });
        
        control_future.wait();
        steps++;
        
        // Log every 200ms
        if (steps % 200 == 0) {
            std::cout << "  t=" << t << "s, angle=" << pendulum_state[0] 
                      << " rad, velocity=" << pendulum_state[1] << " rad/s\n";
        }
    }
    
    end_time_chrono = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_chrono - start_time);
    
    std::cout << "Control loop completed in " << duration.count() << "ms\n";
    std::cout << "Average loop time: " << (duration.count() * 1000.0) / steps << " microseconds\n";
    std::cout << "Final pendulum state: angle=" << pendulum_state[0] 
              << " rad, velocity=" << pendulum_state[1] << " rad/s\n";
    
    // Example 4: Parallel pattern usage
    std::cout << "\n4. Parallel Pattern Demonstration\n";
    std::cout << "   Using parallel_map and parallel_reduce with ODE solutions\n\n";
    
    // Create a range of parameters for parametric study
    std::vector<double> damping_values = {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    
    // Parallel map: solve ODE for each damping value
    auto final_energies = diffeq::execution::patterns::parallel_map(
        damping_values.begin(), damping_values.end(), 
        [](double damping) {
            // Damped harmonic oscillator
            auto damped_oscillator = [damping](double /*t*/, const std::vector<double>& state, std::vector<double>& derivative) {
                derivative[0] = state[1];                           // dx/dt = v
                derivative[1] = -state[0] - 2.0 * damping * state[1]; // dv/dt = -x - 2γv
            };
            
            auto integrator = diffeq::integrators::ode::RK45Integrator<std::vector<double>, double>(damped_oscillator);
            
            std::vector<double> state = {1.0, 0.0};  // Initial: x=1, v=0
            double t = 0.0;
            const double dt = 0.01;
            const double end_time = 10.0;
            
            while (t < end_time) {
                integrator.step(state, dt);
                t += dt;
            }
            
            // Calculate final energy: E = 0.5 * (x² + v²)
            return 0.5 * (state[0] * state[0] + state[1] * state[1]);
        }
    );
    
    std::cout << "Parallel parametric study results:\n";
    for (size_t i = 0; i < damping_values.size(); ++i) {
        std::cout << "  Damping γ=" << damping_values[i] 
                  << ", Final energy=" << final_energies[i] << "\n";
    }
    
    // Parallel reduce: calculate total energy dissipated
    double total_energy_dissipated = diffeq::execution::patterns::parallel_reduce(
        final_energies.begin(), final_energies.end(), 
        0.0, 
        [](double acc, double energy) { return acc + (0.5 - energy); }  // Initial energy was 0.5
    );
    
    std::cout << "Total energy dissipated across all cases: " << total_energy_dissipated << "\n";
    
    std::cout << "\n=== Enhanced Parallelism Integration Complete ===\n";
    std::cout << "The parallelism facade seamlessly integrates with existing diffeq integrators,\n";
    std::cout << "providing transparent acceleration for various computational patterns.\n";
    
    return 0;
}