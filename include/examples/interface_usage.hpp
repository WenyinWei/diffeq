#pragma once

#include <interfaces/integration_interface.hpp>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <chrono>
#include <iostream>

namespace diffeq::examples {

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
    auto interface = interfaces::make_integration_interface<StateType>();
    
    // Example: Market data signal causes continuous trajectory shift
    interface->template register_signal_influence<double>(
        "price_update",
        interfaces::IntegrationInterface<StateType>::InfluenceMode::CONTINUOUS_SHIFT,
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
        interfaces::IntegrationInterface<StateType>::InfluenceMode::DISCRETE_EVENT,
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
    auto interface = interfaces::make_integration_interface<StateType>();
    
    // Example: Control command causes discrete position target update
    interface->template register_signal_influence<std::vector<double>>(
        "control_command",
        interfaces::IntegrationInterface<StateType>::InfluenceMode::DISCRETE_EVENT,
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
        interfaces::IntegrationInterface<StateType>::InfluenceMode::DISCRETE_EVENT,
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
    auto interface = interfaces::make_integration_interface<StateType>();
    
    // Example: Parameter update from external optimization
    interface->template register_signal_influence<double>(
        "parameter_update",
        interfaces::IntegrationInterface<StateType>::InfluenceMode::PARAMETER_UPDATE,
        [](const double& new_param, StateType& state, auto t) {
            // Update system parameters affecting dynamics
            // This could modify integration tolerances, system constants, etc.
        }
    );
    
    // Example: Data logging triggered by external events
    interface->template register_signal_influence<std::string>(
        "log_trigger",
        interfaces::IntegrationInterface<StateType>::InfluenceMode::OUTPUT_TRIGGER,
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
    auto integrator = diffeq::make_rk45<StateType>(signal_aware_ode);
    
    // Simulate some signals during integration
    auto signal_proc = interface->get_signal_processor();
    
    StateType state = initial_state;
    
    // Integration with signal processing
    for (int step = 0; step < 100; ++step) {
        integrator.integrate(state, 0.01, 0.01 * step);
        
        // Simulate external signals
        if (step == 25) {
            signal_proc->emit_signal("price_update", 105.0);
        }
        if (step == 50) {
            signal_proc->emit_signal("risk_alert", std::string("high_volatility"));
        }
        if (step == 75) {
            signal_proc->emit_signal("price_update", 95.0);
        }
    }
}

} // namespace diffeq::examples
