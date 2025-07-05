#include <interfaces/integration_interface.hpp>
#include <diffeq.hpp>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <chrono>
#include <iostream>

/**
 * @brief Example usage patterns for the general integration interface
 * 
 * These examples show how the unified interface can handle various
 * application domains (finance, robotics, etc.) without domain-specific code.
 */

/**
 * @brief Financial portfolio example using the general interface
 */
template<system_state StateType>
auto create_portfolio_interface() {
    auto interface = diffeq::interfaces::make_integration_interface<StateType, double>();
    
    // Example: Market data signal causes continuous trajectory shift
    interface->template register_signal_influence<double>(
        "price_update",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::CONTINUOUS_SHIFT,
        [](const double& new_price, StateType& state, auto t) {
            // Modify portfolio dynamics based on price update
            if (state.size() >= 3) {
                // Simple momentum adjustment
                double momentum = (new_price > 100.0) ? 0.01 : -0.01;
                for (size_t i = 0; i < 3 && i < state.size(); ++i) {
                    state[i] *= (1.0 + momentum);
                }
            }
        }
    );
    
    // Example: Risk alert causes discrete state modification
    interface->template register_signal_influence<std::string>(
        "risk_alert",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::DISCRETE_EVENT,
        [](const std::string& alert_type, StateType& state, auto t) {
            if (alert_type == "high_volatility" && state.size() >= 3) {
                // Reduce all positions by 10%
                for (size_t i = 0; i < 3 && i < state.size(); ++i) {
                    state[i] *= 0.9;
                }
            }
        }
    );
    
    // Example: Real-time portfolio value output
    interface->register_output_stream(
        "portfolio_monitor",
        [](const StateType& state, auto t) {
            if (state.size() >= 3) {
                double total_value = 0.0;
                for (size_t i = 0; i < 3; ++i) {
                    total_value += state[i];
                }
                std::cout << "Portfolio value at t=" << t << ": $" << total_value << std::endl;
            }
        },
        std::chrono::milliseconds{100}  // Update every 100ms
    );
    
    return interface;
}

/**
 * @brief Robotics control example using the general interface
 */
template<system_state StateType>
auto create_robotics_interface() {
    auto interface = diffeq::interfaces::make_integration_interface<StateType, double>();
    
    // Example: Control command causes discrete position target update
    interface->template register_signal_influence<std::vector<double>>(
        "control_command",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::DISCRETE_EVENT,
        [](const std::vector<double>& targets, StateType& state, auto t) {
            // Update target positions (assuming state has position targets)
            size_t n_joints = std::min(targets.size(), state.size() / 3);
            for (size_t i = 0; i < n_joints; ++i) {
                // In a full implementation, would update target positions
                // Here we just show the pattern
                if (i * 3 + 2 < state.size()) {
                    state[i * 3 + 2] = targets[i]; // Simplified target update
                }
            }
        }
    );
    
    // Example: Emergency stop signal
    interface->template register_signal_influence<bool>(
        "emergency_stop",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::DISCRETE_EVENT,
        [](const bool& stop, StateType& state, auto t) {
            if (stop) {
                // Set all velocities to zero
                size_t n_joints = state.size() / 3;
                for (size_t i = 0; i < n_joints; ++i) {
                    if (i + n_joints < state.size()) {
                        state[i + n_joints] = 0.0; // Zero velocity
                    }
                }
            }
        }
    );
    
    // Example: Real-time joint state monitoring
    interface->register_output_stream(
        "joint_monitor",
        [](const StateType& state, auto t) {
            size_t n_joints = state.size() / 3;
            std::cout << "Joint states at t=" << t << ": ";
            for (size_t i = 0; i < n_joints; ++i) {
                if (i < state.size()) {
                    std::cout << "q" << i << "=" << state[i] << " ";
                }
            }
            std::cout << std::endl;
        },
        std::chrono::milliseconds{50}  // High-frequency monitoring
    );
    
    return interface;
}

/**
 * @brief General scientific simulation example
 */
template<system_state StateType>
auto create_scientific_interface() {
    auto interface = diffeq::interfaces::make_integration_interface<StateType, double>();
    
    // Example: Parameter update from external optimization
    interface->template register_signal_influence<double>(
        "parameter_update",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::PARAMETER_UPDATE,
        [](const double& new_param, StateType& state, auto t) {
            // Update system parameters affecting dynamics
            // This could modify integration tolerances, system constants, etc.
        }
    );
    
    // Example: Data logging triggered by external events
    interface->template register_signal_influence<std::string>(
        "log_trigger",
        diffeq::interfaces::IntegrationInterface<StateType, double>::InfluenceMode::OUTPUT_TRIGGER,
        [](const std::string& log_type, StateType& state, auto t) {
            // This will trigger all output streams immediately
        }
    );
    
    // Example: Continuous data export
    interface->register_output_stream(
        "data_export",
        [](const StateType& state, auto t) {
            // Export state data to file, network, database, etc.
            // Implementation depends on user's requirements
        },
        std::chrono::seconds{1}  // Export every second
    );
    
    return interface;
}

/**
 * @brief Usage example showing how to integrate everything
 */
template<system_state StateType>
void demonstrate_usage(StateType initial_state) {
    std::cout << "\n=== Integration Interface Usage Demo ===" << std::endl;
    
    // Create interface for your domain (finance example)
    auto interface = create_portfolio_interface<StateType>();
    
    // Define your ODE system
    auto portfolio_ode = [](auto t, const StateType& y, StateType& dydt) {
        // Basic portfolio dynamics
        for (size_t i = 0; i < y.size() && i < 3; ++i) {
            dydt[i] = 0.05 * y[i]; // 5% annual growth
        }
        // Add risk and other metrics as needed
    };
    
    // Wrap ODE with signal awareness
    auto signal_aware_ode = interface->make_signal_aware_ode(portfolio_ode);
    
    // Create integrator and run
    auto integrator = std::make_unique<diffeq::integrators::ode::RK45Integrator<StateType>>(signal_aware_ode);
    
    // Simulate some signals during integration
    auto signal_proc = interface->get_signal_processor();
    
    StateType state = initial_state;
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    
    std::cout << "Initial portfolio state: [" << state[0] << ", " << state[1] << ", " << state[2] << "]" << std::endl;
    
    // Integrate with signal processing
    for (double t = t_start; t < t_end; t += dt) {
        // Simulate market signals
        if (t > 0.3 && t < 0.4) {
            signal_proc->emit_signal("price_update", 110.0);
        }
        if (t > 0.7) {
            signal_proc->emit_signal("risk_alert", std::string("high_volatility"));
        }
        
        // Integrate one step
        integrator->step(state, dt);
    }
    
    std::cout << "Final portfolio state: [" << state[0] << ", " << state[1] << ", " << state[2] << "]" << std::endl;
    std::cout << "Integration completed successfully!" << std::endl;
}

int main() {
    std::cout << "=== diffeq Integration Interface Examples ===" << std::endl;
    
    // Example with vector state
    std::vector<double> initial_portfolio = {1000.0, 2000.0, 1500.0};
    demonstrate_usage(initial_portfolio);
    
    std::cout << "\n=== Robotics Interface Example ===" << std::endl;
    
    // Create robotics interface
    auto robotics_interface = create_robotics_interface<std::vector<double>>();
    
    // Define robot dynamics
    auto robot_ode = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        // Simple double integrator: d²θ/dt² = u
        dydt[0] = y[1];  // dθ/dt = ω
        dydt[1] = -0.1 * y[0] - 0.5 * y[1];  // Simple PD control
    };
    
    auto signal_aware_robot_ode = robotics_interface->make_signal_aware_ode(robot_ode);
    auto robot_integrator = std::make_unique<diffeq::integrators::ode::RK45Integrator<std::vector<double>>>(signal_aware_robot_ode);
    
    std::vector<double> robot_state = {0.1, 0.0}; // [angle, angular_velocity]
    auto robot_signal_proc = robotics_interface->get_signal_processor();
    
    std::cout << "Initial robot state: angle=" << robot_state[0] << " rad, velocity=" << robot_state[1] << " rad/s" << std::endl;
    
    // Simulate robot control
    for (double t = 0.0; t < 2.0; t += 0.01) {
        if (t > 0.5) {
            robot_signal_proc->emit_signal("control_command", std::vector<double>{0.5});
        }
        if (t > 1.5) {
            robot_signal_proc->emit_signal("emergency_stop", true);
        }
        
        robot_integrator->step(robot_state, 0.01);
    }
    
    std::cout << "Final robot state: angle=" << robot_state[0] << " rad, velocity=" << robot_state[1] << " rad/s" << std::endl;
    
    std::cout << "\n=== Scientific Interface Example ===" << std::endl;
    
    // Create scientific interface
    auto scientific_interface = create_scientific_interface<std::vector<double>>();
    
    // Define scientific system (e.g., chemical reaction)
    auto chemical_ode = [](double t, const std::vector<double>& y, std::vector<double>& dydt) {
        // Simple chemical reaction: A -> B
        double k = 0.1; // reaction rate
        dydt[0] = -k * y[0];  // dA/dt = -k*A
        dydt[1] = k * y[0];   // dB/dt = k*A
    };
    
    auto signal_aware_chemical_ode = scientific_interface->make_signal_aware_ode(chemical_ode);
    auto chemical_integrator = std::make_unique<diffeq::integrators::ode::RK45Integrator<std::vector<double>>>(signal_aware_chemical_ode);
    
    std::vector<double> chemical_state = {1.0, 0.0}; // [A, B]
    auto chemical_signal_proc = scientific_interface->get_signal_processor();
    
    std::cout << "Initial chemical state: A=" << chemical_state[0] << ", B=" << chemical_state[1] << std::endl;
    
    // Simulate chemical reaction with parameter updates
    for (double t = 0.0; t < 10.0; t += 0.1) {
        if (t > 5.0) {
            chemical_signal_proc->emit_signal("parameter_update", 0.2); // Increase reaction rate
        }
        
        chemical_integrator->step(chemical_state, 0.1);
    }
    
    std::cout << "Final chemical state: A=" << chemical_state[0] << ", B=" << chemical_state[1] << std::endl;
    
    std::cout << "\n=== All examples completed successfully! ===" << std::endl;
    
    return 0;
} 