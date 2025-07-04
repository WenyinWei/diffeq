#include <async/async_integrator.hpp>
#include <signal/signal_processor.hpp>
#include <domains/application_processors.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>

using namespace diffeq;
using namespace diffeq::async;
using namespace diffeq::signal;
using namespace diffeq::domains;

/**
 * @brief Quantitative Finance Example: Real-time Portfolio Optimization
 * 
 * This example demonstrates how to use the diffeq library for real-time
 * portfolio optimization in quantitative trading. The ODE system models
 * portfolio dynamics while processing real-time market signals.
 */
void quantitative_finance_example() {
    std::cout << "=== Quantitative Finance Portfolio Optimization ===\n";
    
    // Portfolio state: [AAPL_value, GOOGL_value, MSFT_value, portfolio_VaR, sharpe_ratio]
    std::vector<double> portfolio_state = {100000.0, 150000.0, 120000.0, 5000.0, 1.2};
    
    // Create finance processor
    auto integrator = factory::make_async_dop853<std::vector<double>>(
        [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
            // Placeholder ODE - will be set by finance processor
            std::fill(dydt.begin(), dydt.end(), 0.0);
        },
        AsyncIntegrator<std::vector<double>>::Config{
            .enable_async_stepping = true,
            .enable_state_monitoring = true
        },
        1e-8,  // High precision for financial calculations
        1e-12
    );
    
    auto signal_processor = make_signal_processor(integrator);
    auto finance_processor = factory::make_finance_processor(signal_processor);
    
    // Set up the actual portfolio dynamics
    integrator->set_system([&finance_processor](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        finance_processor->portfolio_dynamics(t, y, dydt);
    });
    
    // Start async operation
    integrator->start();
    
    // Simulate real-time market data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> price_dist(0.0, 0.01); // 1% volatility
    
    std::vector<std::string> symbols = {"AAPL", "GOOGL", "MSFT"};
    std::vector<double> base_prices = {150.0, 2800.0, 350.0};
    
    std::cout << "Starting portfolio optimization...\n";
    std::cout << "Time(s)\tAAPL($)\tGOOGL($)\tMSFT($)\tValue($)\n";
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (int step = 0; step < 100; ++step) {
        // Generate market signals
        for (size_t i = 0; i < symbols.size(); ++i) {
            MarketData market_data;
            market_data.symbol = symbols[i];
            market_data.price = base_prices[i] * (1.0 + price_dist(gen));
            market_data.volume = 1000.0 + std::abs(price_dist(gen)) * 5000.0;
            market_data.volatility = 0.2 + price_dist(gen) * 0.1;
            market_data.timestamp = std::chrono::steady_clock::now();
            
            // Send market data signal
            auto signal = Signal<MarketData>{market_data, "market_data"};
            signal_processor->process_signal(signal);
        }
        
        // Integrate for 0.1 second
        double dt = 0.01;
        double end_time = step * 0.1;
        integrator->integrate(portfolio_state, dt, end_time);
        
        // Update finance processor with ODE results
        finance_processor->update_from_ode_state(portfolio_state);
        
        // Display results every 10 steps
        if (step % 10 == 0) {
            auto summary = finance_processor->get_summary();
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time
            ).count();
            
            std::cout << std::fixed << std::setprecision(2)
                      << elapsed << "\t"
                      << portfolio_state[0] << "\t"
                      << portfolio_state[1] << "\t" 
                      << portfolio_state[2] << "\t"
                      << summary.total_value << "\n";
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Portfolio optimization complete.\n";
    auto final_summary = finance_processor->get_summary();
    std::cout << "Final Portfolio Value: $" << final_summary.total_value << "\n";
    std::cout << "Unrealized P&L: $" << final_summary.unrealized_pnl << "\n\n";
}

/**
 * @brief Robotics Example: Real-time Robot Arm Control
 * 
 * This example shows real-time control of a 6-DOF robot arm using
 * ODE integration for dynamics and real-time signal processing
 * for control commands.
 */
void robotics_control_example() {
    std::cout << "=== Robotics 6-DOF Robot Arm Control ===\n";
    
    constexpr size_t N_JOINTS = 6;
    using RobotState = std::array<double, N_JOINTS * 3>; // pos, vel, acc for each joint
    
    // Initial robot state: all joints at zero position
    RobotState robot_state{};
    
    // Robot configuration
    RobotConfig robot_config;
    robot_config.link_masses = {2.0, 3.0, 1.5, 1.0, 0.8, 0.3};     // kg
    robot_config.link_lengths = {0.3, 0.4, 0.3, 0.2, 0.15, 0.1};   // m
    robot_config.joint_damping = {0.1, 0.15, 0.1, 0.05, 0.03, 0.02}; // N⋅m⋅s/rad
    robot_config.max_torque = {100, 80, 60, 40, 20, 10};            // N⋅m
    robot_config.max_velocity = {2.0, 1.5, 2.0, 3.0, 3.0, 4.0};    // rad/s
    
    // Control gains (PID)
    RoboticsProcessor<N_JOINTS>::ControlGains gains;
    gains.kp = {100, 120, 80, 60, 40, 30};   // Position gains
    gains.kd = {20, 25, 15, 10, 8, 5};       // Derivative gains  
    gains.ki = {1, 1, 1, 1, 1, 1};           // Integral gains
    
    // Create async integrator
    auto integrator = factory::make_async_rk45<RobotState>(
        [](double t, const RobotState& y, RobotState& dydt) {
            // Placeholder - will be set by robotics processor
            std::fill(dydt.begin(), dydt.end(), 0.0);
        },
        AsyncIntegrator<RobotState>::Config{
            .enable_async_stepping = true,
            .enable_state_monitoring = true,
            .monitoring_interval = std::chrono::microseconds(50)
        },
        1e-6,  // Good balance of speed and accuracy for control
        1e-9
    );
    
    auto signal_processor = make_signal_processor(integrator);
    auto robot_processor = factory::make_robotics_processor<N_JOINTS>(
        signal_processor, robot_config, gains);
    
    // Set up robot dynamics
    integrator->set_system([&robot_processor](double t, const RobotState& y, RobotState& dydt) {
        robot_processor->robot_dynamics(t, y, dydt);
    });
    
    // Start async operation
    integrator->start();
    
    std::cout << "Starting robot control...\n";
    std::cout << "Time(s)\tJ1(°)\tJ2(°)\tJ3(°)\tJ4(°)\tJ5(°)\tJ6(°)\n";
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Simulate trajectory following
    for (int step = 0; step < 200; ++step) {
        double t = step * 0.005; // 200Hz control loop
        
        // Generate sinusoidal trajectory for demonstration
        std::vector<double> control_targets = {
            0.5 * std::sin(0.5 * t),        // Joint 1: slow sine wave
            0.3 * std::sin(0.8 * t + 1.0),  // Joint 2: offset sine
            0.4 * std::sin(1.2 * t + 2.0),  // Joint 3: faster sine
            0.2 * std::sin(2.0 * t),        // Joint 4: fast sine
            0.3 * std::sin(1.5 * t + 0.5),  // Joint 5: medium sine
            0.1 * std::sin(3.0 * t + 1.5)   // Joint 6: very fast sine
        };
        
        // Send control signal
        auto signal = Signal<std::vector<double>>{control_targets, "control_targets", 5.0};
        signal_processor->process_signal(signal);
        
        // Integrate dynamics for one control step
        double dt = 0.001;
        double end_time = t + 0.005;
        integrator->integrate(robot_state, dt, end_time);
        
        // Update robot processor with ODE results
        robot_processor->update_from_ode_state(robot_state);
        
        // Display results every 20 steps (10Hz display)
        if (step % 20 == 0) {
            const auto& joints = robot_processor->get_joint_states();
            
            std::cout << std::fixed << std::setprecision(3)
                      << t << "\t";
                      
            for (size_t i = 0; i < N_JOINTS; ++i) {
                std::cout << joints[i].position * 180.0 / M_PI << "\t"; // Convert to degrees
            }
            std::cout << "\n";
        }
        
        // Simulate emergency stop at step 150
        if (step == 150) {
            std::cout << "*** EMERGENCY STOP TRIGGERED ***\n";
            integrator->emergency_stop();
        }
        
        std::this_thread::sleep_for(std::chrono::microseconds(5000)); // 200Hz
    }
    
    std::cout << "Robot control simulation complete.\n";
    
    // Display final joint states
    const auto& final_joints = robot_processor->get_joint_states();
    std::cout << "Final Joint Positions (degrees):\n";
    for (size_t i = 0; i < N_JOINTS; ++i) {
        std::cout << "  Joint " << (i+1) << ": " 
                  << std::fixed << std::setprecision(2)
                  << final_joints[i].position * 180.0 / M_PI << "°\n";
    }
    std::cout << "\n";
}

/**
 * @brief Stochastic Differential Equation Example
 * 
 * Demonstrates handling stochastic systems with noise for both
 * financial modeling and robot control under uncertainty.
 */
void stochastic_differential_equation_example() {
    std::cout << "=== Stochastic Differential Equations with Noise ===\n";
    
    // Geometric Brownian Motion for stock price modeling
    // dS = μS dt + σS dW
    std::vector<double> stock_state = {100.0}; // Initial stock price
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise(0.0, 1.0);
    
    auto gbm_with_noise = [&noise, &gen](
        double t,
        const std::vector<double>& y,
        std::vector<double>& dydt
    ) {
        double mu = 0.05;    // 5% drift
        double sigma = 0.2;  // 20% volatility
        double dt = 0.001;   // Time step for noise
        
        // Deterministic part: μS
        dydt[0] = mu * y[0];
        
        // Add stochastic part: σS * dW/dt ≈ σS * ξ/√dt
        double dW = noise(gen) * std::sqrt(dt);
        dydt[0] += sigma * y[0] * dW / dt;
    };
    
    // Create integrator for SDE
    RealtimeIntegrator<std::vector<double>>::RealtimeConfig sde_config;
    sde_config.signal_processing_interval = std::chrono::microseconds(100);
    
    auto sde_integrator = factory::make_realtime_rk45<std::vector<double>>(
        gbm_with_noise,
        sde_config
    );
    
    std::cout << "Simulating Geometric Brownian Motion (Stock Price):\n";
    std::cout << "Time(s)\tPrice($)\n";
    
    for (int step = 0; step < 100; ++step) {
        double dt = 0.01;
        double end_time = step * 0.1;
        
        sde_integrator->integrate(stock_state, dt, end_time);
        
        if (step % 10 == 0) {
            std::cout << std::fixed << std::setprecision(3)
                      << end_time << "\t" << stock_state[0] << "\n";
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    std::cout << "Final stock price: $" << std::fixed << std::setprecision(2) 
              << stock_state[0] << "\n\n";
}

int main() {
    std::cout << "DiffEq Real-time Signal Processing Examples\n";
    std::cout << "==========================================\n\n";
    
    try {
        // Run quantitative finance example
        quantitative_finance_example();
        
        // Run robotics control example  
        robotics_control_example();
        
        // Run stochastic differential equation example
        stochastic_differential_equation_example();
        
        std::cout << "All examples completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
