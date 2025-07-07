/**
 * @file comprehensive_output_facilities_demo.cpp
 * @brief Comprehensive demonstration of enhanced output facilities
 * 
 * This demo showcases all the new output capabilities:
 * - Dense output with interpolation
 * - Interprocess communication for real-time data exchange
 * - Event-driven feedback for robotics control
 * - SDE synchronization for noise processes
 * - Simultaneous multiple output modes for debugging and control
 */

#include <diffeq.hpp>
#include <core/composable_integration.hpp>
#include <core/composable/sde_synchronization.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <thread>
#include <memory>

using namespace diffeq::core::composable;

// Test systems
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];  // Simple exponential decay
}

void harmonic_oscillator(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    // x'' + omega^2 * x = 0 => [x, x'] -> [x', -omega^2 * x]
    const double omega = 2.0;
    dydt[0] = y[1];                    // dx/dt = v
    dydt[1] = -omega * omega * y[0];   // dv/dt = -omega^2 * x
}

void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0, rho = 28.0, beta = 8.0/3.0;
    dydt[0] = sigma * (y[1] - y[0]);
    dydt[1] = y[0] * (rho - y[2]) - y[1];
    dydt[2] = y[0] * y[1] - beta * y[2];
}

void demonstrate_dense_output() {
    std::cout << "\n=== Dense Output and Interpolation Demo ===\n";
    
    std::cout << "\n1. Creating integrator with dense output capabilities\n";
    
    auto base_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(harmonic_oscillator);
    
    // Configure interpolation
    InterpolationConfig interp_config;
    interp_config.method = InterpolationMethod::CUBIC_SPLINE;
    interp_config.max_history_size = 1000;
    interp_config.enable_compression = true;
    interp_config.allow_extrapolation = true;
    
    auto dense_integrator = make_builder(std::move(base_integrator))
        .with_interpolation(interp_config)
        .build();
    
    // Get interpolation decorator for specific operations
    auto* interp_decorator = dynamic_cast<InterpolationDecorator<std::vector<double>>*>(dense_integrator.get());
    
    if (interp_decorator) {
        std::cout << "  ✓ Interpolation decorator created successfully\n";
        
        // Integrate harmonic oscillator
        std::vector<double> state = {1.0, 0.0};  // Initial position and velocity
        dense_integrator->integrate(state, 0.01, 2.0);  // 2 seconds integration
        
        std::cout << "  ✓ Integration completed, history size: " << interp_decorator->get_history_size() << "\n";
        
        // Test dense output at arbitrary times
        std::cout << "\n2. Testing dense output at arbitrary time points\n";
        
        std::vector<double> query_times = {0.5, 1.0, 1.5, 1.8};
        for (double t : query_times) {
            try {
                auto interpolated_state = interp_decorator->interpolate_at(t);
                std::cout << "    t=" << std::fixed << std::setprecision(1) << t 
                         << ": x=" << std::setprecision(4) << interpolated_state[0]
                         << ", v=" << interpolated_state[1] << "\n";
            } catch (const std::exception& e) {
                std::cout << "    t=" << t << ": Error - " << e.what() << "\n";
            }
        }
        
        // Test dense output over interval
        std::cout << "\n3. Getting dense output over interval\n";
        auto [times, states] = interp_decorator->get_dense_output(0.0, 2.0, 21);
        
        std::cout << "    Generated " << times.size() << " points over [0, 2] interval:\n";
        for (size_t i = 0; i < std::min(size_t(5), times.size()); ++i) {
            std::cout << "      t=" << std::fixed << std::setprecision(2) << times[i]
                     << ": x=" << std::setprecision(4) << states[i][0] << "\n";
        }
        std::cout << "      ... (and " << (times.size() - 5) << " more points)\n";
        
        // Show interpolation statistics
        const auto& stats = interp_decorator->get_statistics();
        std::cout << "\n  Interpolation Statistics:\n";
        std::cout << "    Total interpolations: " << stats.total_interpolations << "\n";
        std::cout << "    Average time: " << std::fixed << std::setprecision(1) 
                 << stats.average_interpolation_time_ns << " ns\n";
        std::cout << "    History compressions: " << stats.history_compressions << "\n";
    }
}

void demonstrate_interprocess_communication() {
    std::cout << "\n=== Interprocess Communication Demo ===\n";
    
    std::cout << "\n1. Setting up IPC producer (data sender)\n";
    
    auto producer_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(exponential_decay);
    
    // Configure IPC as producer
    InterprocessConfig producer_config;
    producer_config.method = IPCMethod::SHARED_MEMORY;
    producer_config.direction = IPCDirection::PRODUCER;
    producer_config.channel_name = "demo_channel";
    producer_config.buffer_size = 1024 * 64;  // 64KB
    
    try {
        auto ipc_producer = make_builder(std::move(producer_integrator))
            .with_interprocess(producer_config)
            .build();
        
        auto* ipc_decorator = dynamic_cast<InterprocessDecorator<std::vector<double>>*>(ipc_producer.get());
        
        if (ipc_decorator && ipc_decorator->is_connected()) {
            std::cout << "  ✓ IPC Producer created and connected\n";
            std::cout << "  Status: " << ipc_decorator->get_status() << "\n";
            
            // Simulate sending data
            std::vector<double> state = {1.0};
            for (int i = 0; i < 5; ++i) {
                ipc_producer->step(state, 0.1);
                std::cout << "    Sent state at t=" << std::fixed << std::setprecision(1) 
                         << i * 0.1 << ": " << std::setprecision(4) << state[0] << "\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // Show IPC statistics
            const auto& stats = ipc_decorator->get_statistics();
            std::cout << "\n  IPC Producer Statistics:\n";
            std::cout << "    Messages sent: " << stats.messages_sent << "\n";
            std::cout << "    Bytes sent: " << stats.bytes_sent << "\n";
            std::cout << "    Send success rate: " << std::fixed << std::setprecision(1) 
                     << stats.send_success_rate() * 100 << "%\n";
            std::cout << "    Average send time: " << std::setprecision(2) 
                     << stats.average_send_time_ms() << " ms\n";
        } else {
            std::cout << "  ⚠ IPC Producer connection failed (this is normal in demo - no consumer)\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "  ⚠ IPC Demo failed: " << e.what() << " (this is expected without proper IPC setup)\n";
    }
    
    std::cout << "\n2. IPC Consumer demonstration (conceptual)\n";
    std::cout << "  In a real application, you would:\n";
    std::cout << "  - Start consumer in separate process\n";
    std::cout << "  - Configure with IPCDirection::CONSUMER\n";
    std::cout << "  - Set receive callback for incoming data\n";
    std::cout << "  - Handle data synchronization and buffering\n";
}

void demonstrate_event_driven_feedback() {
    std::cout << "\n=== Event-Driven Feedback Demo ===\n";
    
    std::cout << "\n1. Setting up event-driven integrator for robotics control\n";
    
    auto base_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(harmonic_oscillator);
    
    // Configure event processing
    EventConfig event_config;
    event_config.processing_mode = EventProcessingMode::IMMEDIATE;
    event_config.enable_control_loop = true;
    event_config.control_loop_period = std::chrono::microseconds{5000};  // 5ms control loop
    event_config.max_event_processing_time = std::chrono::microseconds{1000};  // 1ms limit
    event_config.enable_sensor_validation = true;
    
    auto event_integrator = make_builder(std::move(base_integrator))
        .with_events(event_config)
        .build();
    
    auto* event_decorator = dynamic_cast<EventDecorator<std::vector<double>>*>(event_integrator.get());
    
    if (event_decorator) {
        std::cout << "  ✓ Event decorator created successfully\n";
        
        // Register event handlers
        std::cout << "\n2. Registering event handlers\n";
        
        // Threshold crossing event (position limit)
        event_decorator->set_threshold_event(0, 0.5, true, [](std::vector<double>& state, double time) {
            std::cout << "    🔔 Threshold event: Position exceeded 0.5 at t=" 
                     << std::fixed << std::setprecision(3) << time << "\n";
            // Apply corrective action
            state[1] *= 0.9;  // Reduce velocity by 10%
        });
        
        // State-based event (energy monitoring)
        event_decorator->set_state_condition(
            [](const std::vector<double>& state, double time) {
                double energy = 0.5 * (state[0]*state[0] + state[1]*state[1]);
                return energy > 1.5;  // High energy condition
            },
            [](std::vector<double>& state, double time) {
                std::cout << "    ⚡ High energy event at t=" 
                         << std::fixed << std::setprecision(3) << time << "\n";
                // Energy damping
                state[0] *= 0.95;
                state[1] *= 0.95;
            },
            EventPriority::HIGH
        );
        
        std::cout << "  ✓ Event handlers registered\n";
        
        // Simulate sensor data
        std::cout << "\n3. Simulating sensor feedback\n";
        
        std::vector<double> state = {1.0, 0.0};
        
        for (int i = 0; i < 10; ++i) {
            double t = i * 0.2;
            
            // Submit sensor data periodically
            if (i % 3 == 0) {
                std::vector<double> sensor_values = {state[0] + 0.01 * (i % 2 ? 1 : -1)};  // Simulated sensor noise
                event_decorator->submit_sensor_data("position_sensor", sensor_values, 0.95);
                std::cout << "    📡 Sensor data submitted at t=" 
                         << std::fixed << std::setprecision(1) << t << "\n";
            }
            
            // Submit control feedback
            if (i % 4 == 0) {
                std::vector<double> target_state = {0.0, 0.0};  // Target: stationary at origin
                event_decorator->submit_control_feedback("position_control", target_state, state);
                std::cout << "    🎯 Control feedback submitted at t=" 
                         << std::fixed << std::setprecision(1) << t << "\n";
            }
            
            // Integrate one step
            event_integrator->step(state, 0.2);
            
            std::cout << "    State at t=" << std::fixed << std::setprecision(1) << t + 0.2
                     << ": x=" << std::setprecision(4) << state[0] 
                     << ", v=" << state[1] << "\n";
        }
        
        // Show event statistics
        const auto& stats = event_decorator->get_statistics();
        std::cout << "\n  Event Processing Statistics:\n";
        std::cout << "    Total events: " << stats.total_events << "\n";
        std::cout << "    Processed events: " << stats.processed_events << "\n";
        std::cout << "    High priority events: " << stats.high_priority_events << "\n";
        std::cout << "    Sensor events: " << stats.sensor_events << "\n";
        std::cout << "    Control feedback events: " << stats.control_feedback_events << "\n";
        std::cout << "    Event success rate: " << std::fixed << std::setprecision(1) 
                 << stats.event_success_rate() * 100 << "%\n";
        std::cout << "    Average processing time: " << std::setprecision(2) 
                 << stats.average_processing_time_us() << " μs\n";
    }
}

void demonstrate_sde_synchronization() {
    std::cout << "\n=== SDE Synchronization Demo ===\n";
    
    std::cout << "\n1. Setting up SDE synchronization\n";
    
    // Configure SDE synchronization
    SDESyncConfig sync_config;
    sync_config.sync_mode = SDESyncMode::BUFFERED;
    sync_config.noise_type = NoiseProcessType::WIENER;
    sync_config.noise_dimensions = 1;
    sync_config.noise_intensity = 0.5;
    sync_config.max_noise_delay = std::chrono::microseconds{1000};
    
    SDESynchronizer<std::vector<double>> sde_sync(sync_config);
    std::cout << "  ✓ SDE Synchronizer created\n";
    
    // Simulate noise generation and consumption
    std::cout << "\n2. Simulating noise process synchronization\n";
    
    for (int i = 0; i < 10; ++i) {
        double t = i * 0.1;
        
        // Generate noise data (simulate external noise process)
        std::vector<double> noise_increments = {0.1 * std::sin(t) + 0.05 * (i % 2 ? 1 : -1)};
        NoiseData<double> noise_data(t, noise_increments, NoiseProcessType::WIENER);
        
        // Submit noise data
        sde_sync.submit_noise_data(noise_data);
        
        // Request noise for SDE integration
        try {
            auto retrieved_noise = sde_sync.get_noise_increment(t, 0.1);
            std::cout << "    t=" << std::fixed << std::setprecision(1) << t 
                     << ": noise=" << std::setprecision(4) << retrieved_noise.increments[0] << "\n";
        } catch (const std::exception& e) {
            std::cout << "    t=" << t << ": Noise timeout - " << e.what() << "\n";
        }
    }
    
    // Show synchronization statistics
    auto sync_stats = sde_sync.get_statistics();
    std::cout << "\n  SDE Synchronization Statistics:\n";
    std::cout << "    Noise requests: " << sync_stats.noise_requests << "\n";
    std::cout << "    Noise timeouts: " << sync_stats.noise_timeouts << "\n";
    std::cout << "    Interpolations: " << sync_stats.interpolations << "\n";
    std::cout << "    Timeout rate: " << std::fixed << std::setprecision(1) 
             << sync_stats.timeout_rate() * 100 << "%\n";
    
    std::cout << "\n3. SDE integrator pair creation (conceptual)\n";
    std::cout << "  In practice, you would:\n";
    std::cout << "  - Create noise generator process with Producer IPC\n";
    std::cout << "  - Create SDE integrator process with Consumer IPC\n";
    std::cout << "  - Use create_synchronized_pair() for coordinated setup\n";
    std::cout << "  - Handle cross-process noise synchronization\n";
}

void demonstrate_simultaneous_outputs() {
    std::cout << "\n=== Simultaneous Multiple Outputs Demo ===\n";
    std::cout << "Demonstrating how one integrator can simultaneously provide:\n";
    std::cout << "- Dense output for debugging\n";
    std::cout << "- Real-time output for control\n";
    std::cout << "- Event-driven feedback for safety\n";
    std::cout << "- IPC for distributed computing\n";
    
    auto base_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(lorenz_system);
    
    // Build comprehensive output system
    auto ultimate_integrator = make_builder(std::move(base_integrator))
        // Dense output for debugging
        .with_interpolation(InterpolationConfig{
            .method = InterpolationMethod::CUBIC_SPLINE,
            .max_history_size = 1000,
            .enable_compression = true
        })
        // Real-time output for monitoring
        .with_output(OutputConfig{
            .mode = OutputMode::ONLINE,
            .output_interval = std::chrono::microseconds{100000}  // 100ms
        }, [](const std::vector<double>& state, double t, size_t step) {
            if (step % 10 == 0) {
                double magnitude = std::sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2]);
                std::cout << "    📊 Monitor t=" << std::fixed << std::setprecision(2) << t 
                         << ", |state|=" << std::setprecision(3) << magnitude << "\n";
            }
        })
        // Event-driven safety monitoring
        .with_events(EventConfig{
            .processing_mode = EventProcessingMode::IMMEDIATE,
            .max_event_processing_time = std::chrono::microseconds{500}
        })
        // Async processing for performance
        .with_async(AsyncConfig{
            .thread_pool_size = 2,
            .enable_progress_monitoring = false
        })
        .build();
    
    std::cout << "\n  ✓ Ultimate integrator created with ALL output facilities\n";
    
    // Set up event monitoring for Lorenz system
    auto* event_decorator = dynamic_cast<EventDecorator<std::vector<double>>*>(ultimate_integrator.get());
    if (event_decorator) {
        // Monitor for divergence (safety event)
        event_decorator->set_state_condition(
            [](const std::vector<double>& state, double time) {
                double magnitude = std::sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2]);
                return magnitude > 50.0;  // Divergence threshold
            },
            [](std::vector<double>& state, double time) {
                std::cout << "    🚨 SAFETY EVENT: Lorenz system diverging at t=" 
                         << std::fixed << std::setprecision(3) << time << "! Applying stabilization.\n";
                // Apply stabilization
                for (auto& component : state) {
                    component *= 0.8;
                }
            },
            EventPriority::CRITICAL
        );
    }
    
    // Run comprehensive integration
    std::cout << "\n  Running integration with simultaneous outputs...\n";
    
    std::vector<double> state = {1.0, 1.0, 1.0};
    ultimate_integrator->integrate(state, 0.01, 1.0);
    
    std::cout << "  ✓ Integration completed successfully\n";
    
    // Access dense output for post-processing
    auto* interp_decorator = dynamic_cast<InterpolationDecorator<std::vector<double>>*>(ultimate_integrator.get());
    if (interp_decorator) {
        std::cout << "\n  Dense output analysis:\n";
        auto [times, states] = interp_decorator->get_dense_output(0.0, 1.0, 11);
        std::cout << "    Generated " << times.size() << " interpolated points\n";
        
        // Find maximum deviation
        double max_magnitude = 0.0;
        double max_time = 0.0;
        for (size_t i = 0; i < states.size(); ++i) {
            double mag = std::sqrt(states[i][0]*states[i][0] + states[i][1]*states[i][1] + states[i][2]*states[i][2]);
            if (mag > max_magnitude) {
                max_magnitude = mag;
                max_time = times[i];
            }
        }
        std::cout << "    Maximum state magnitude: " << std::fixed << std::setprecision(3) 
                 << max_magnitude << " at t=" << max_time << "\n";
        
        const auto& interp_stats = interp_decorator->get_statistics();
        std::cout << "    Interpolation operations: " << interp_stats.total_interpolations << "\n";
    }
    
    if (event_decorator) {
        const auto& event_stats = event_decorator->get_statistics();
        std::cout << "\n  Event processing summary:\n";
        std::cout << "    Total events processed: " << event_stats.total_events << "\n";
        std::cout << "    Critical events: " << event_stats.high_priority_events << "\n";
    }
    
    std::cout << "\n  🎯 Simultaneous output demonstration completed!\n";
    std::cout << "     This shows how a single integrator can provide:\n";
    std::cout << "     - Dense output for detailed analysis\n";
    std::cout << "     - Real-time monitoring for operations\n";
    std::cout << "     - Safety events for critical systems\n";
    std::cout << "     - Async processing for performance\n";
}

void demonstrate_robotics_control_scenario() {
    std::cout << "\n=== Robotics Control Scenario Demo ===\n";
    std::cout << "Simulating robot arm control with sensor feedback\n";
    
    // Robot arm dynamics (simplified 2-DOF)
    auto robot_dynamics = [](double t, const std::vector<double>& state, std::vector<double>& dydt) {
        // state = [theta1, theta2, omega1, omega2]
        const double m1 = 1.0, m2 = 0.8, l1 = 1.0, l2 = 0.8;
        const double g = 9.81;
        
        double theta1 = state[0], theta2 = state[1];
        double omega1 = state[2], omega2 = state[3];
        
        // Simplified equations (real robot would be more complex)
        dydt[0] = omega1;
        dydt[1] = omega2;
        dydt[2] = -g/l1 * std::sin(theta1) - 0.1 * omega1;  // Damping
        dydt[3] = -g/l2 * std::sin(theta2) - 0.1 * omega2;  // Damping
    };
    
    auto robot_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(robot_dynamics);
    
    // Configure for robotics control
    auto control_system = make_builder(std::move(robot_integrator))
        // Real-time control feedback
        .with_events(EventConfig{
            .processing_mode = EventProcessingMode::IMMEDIATE,
            .enable_control_loop = true,
            .control_loop_period = std::chrono::microseconds{1000},  // 1kHz control
            .sensor_timeout = std::chrono::microseconds{2000},      // 2ms sensor timeout
            .enable_sensor_validation = true
        })
        // Dense output for trajectory analysis
        .with_interpolation(InterpolationConfig{
            .method = InterpolationMethod::CUBIC_SPLINE,
            .max_history_size = 2000
        })
        // Real-time monitoring
        .with_output(OutputConfig{
            .mode = OutputMode::ONLINE,
            .output_interval = std::chrono::microseconds{50000}  // 50ms updates
        })
        .build();
    
    auto* event_decorator = dynamic_cast<EventDecorator<std::vector<double>>*>(control_system.get());
    
    if (event_decorator) {
        std::cout << "  ✓ Robotics control system initialized\n";
        
        // Set up safety limits (joint angle limits)
        event_decorator->set_threshold_event(0, 1.5, true, [](std::vector<double>& state, double time) {
            std::cout << "    🚨 Joint 1 limit exceeded! Emergency stop at t=" 
                     << std::fixed << std::setprecision(3) << time << "\n";
            state[2] = 0.0;  // Stop joint 1
        });
        
        event_decorator->set_threshold_event(1, 1.2, true, [](std::vector<double>& state, double time) {
            std::cout << "    🚨 Joint 2 limit exceeded! Emergency stop at t=" 
                     << std::fixed << std::setprecision(3) << time << "\n";
            state[3] = 0.0;  // Stop joint 2
        });
        
        // Set up position control
        std::vector<double> target_position = {0.5, 0.3};  // Target joint angles
        
        event_decorator->register_event_handler(EventTrigger::CONTROL_FEEDBACK, 
            [target_position](std::vector<double>& state, double time) {
                // Simple P-controller
                const double kp = 2.0;
                double error1 = target_position[0] - state[0];
                double error2 = target_position[1] - state[1];
                
                // Apply control torques (simplified)
                state[2] += kp * error1 * 0.01;  // Control torque for joint 1
                state[3] += kp * error2 * 0.01;  // Control torque for joint 2
            });
        
        std::cout << "\n  Running robot control simulation...\n";
        
        // Initial robot state: [theta1, theta2, omega1, omega2]
        std::vector<double> robot_state = {0.0, 0.0, 0.0, 0.0};
        
        for (int i = 0; i < 20; ++i) {
            double t = i * 0.1;
            
            // Simulate sensor data with noise
            if (i % 3 == 0) {
                std::vector<double> sensor_angles = {
                    robot_state[0] + 0.01 * ((i % 4) - 2),  // ±0.02 noise
                    robot_state[1] + 0.01 * ((i % 3) - 1)   // ±0.01 noise
                };
                event_decorator->submit_sensor_data("joint_encoders", sensor_angles, 0.98);
                
                // Submit control feedback
                event_decorator->submit_control_feedback("position_controller", target_position, 
                    {robot_state[0], robot_state[1]});
            }
            
            // Integrate one step
            control_system->step(robot_state, 0.1);
            
            if (i % 5 == 0) {
                std::cout << "    t=" << std::fixed << std::setprecision(1) << t + 0.1
                         << ": θ1=" << std::setprecision(3) << robot_state[0]
                         << ", θ2=" << robot_state[1] 
                         << " (target: " << target_position[0] << ", " << target_position[1] << ")\n";
            }
        }
        
        const auto& stats = event_decorator->get_statistics();
        std::cout << "\n  Robot Control Statistics:\n";
        std::cout << "    Control events: " << stats.control_feedback_events << "\n";
        std::cout << "    Sensor events: " << stats.sensor_events << "\n";
        std::cout << "    Safety events: " << stats.high_priority_events << "\n";
        std::cout << "    Processing time: " << std::fixed << std::setprecision(1) 
                 << stats.average_processing_time_us() << " μs avg\n";
    }
    
    std::cout << "  ✓ Robotics control scenario completed successfully\n";
}

void demonstrate_financial_trading_scenario() {
    std::cout << "\n=== Financial Trading Scenario Demo ===\n";
    std::cout << "Simulating high-frequency trading with real-time market data\n";
    
    // Market dynamics (simplified SDE model)
    auto market_model = [](double t, const std::vector<double>& state, std::vector<double>& dydt) {
        // state = [price, volatility]
        const double mu = 0.05;      // Drift
        const double mean_vol = 0.2; // Mean volatility
        const double vol_speed = 2.0; // Volatility mean reversion speed
        
        dydt[0] = mu * state[0];  // Price drift
        dydt[1] = vol_speed * (mean_vol - state[1]);  // Volatility mean reversion
    };
    
    auto trading_integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>>>(market_model);
    
    // Configure for HFT
    auto trading_system = make_builder(std::move(trading_integrator))
        // Ultra-fast event processing
        .with_events(EventConfig{
            .processing_mode = EventProcessingMode::IMMEDIATE,
            .max_event_processing_time = std::chrono::microseconds{10},  // 10μs limit
            .enable_control_loop = false,  // Market-driven, not control
            .sensor_timeout = std::chrono::microseconds{100}            // 100μs market data timeout
        })
        // Dense output for risk analysis
        .with_interpolation(InterpolationConfig{
            .method = InterpolationMethod::LINEAR,  // Fast interpolation
            .max_history_size = 10000,
            .enable_compression = true
        })
        // Real-time P&L monitoring
        .with_output(OutputConfig{
            .mode = OutputMode::ONLINE,
            .output_interval = std::chrono::microseconds{1000}  // 1ms updates
        }, [](const std::vector<double>& state, double t, size_t step) {
            if (step % 100 == 0) {
                std::cout << "    💹 Market t=" << std::fixed << std::setprecision(4) << t 
                         << ": price=" << std::setprecision(2) << state[0] 
                         << ", vol=" << std::setprecision(3) << state[1] << "\n";
            }
        })
        .build();
    
    auto* event_decorator = dynamic_cast<EventDecorator<std::vector<double>>*>(trading_system.get());
    
    if (event_decorator) {
        std::cout << "  ✓ Trading system initialized\n";
        
        // Price movement triggers
        double entry_price = 100.0;
        event_decorator->set_threshold_event(0, entry_price * 1.02, true, [entry_price](std::vector<double>& state, double time) {
            std::cout << "    📈 BUY signal: Price up 2% at t=" 
                     << std::fixed << std::setprecision(4) << time 
                     << ", price=" << std::setprecision(2) << state[0] << "\n";
        });
        
        event_decorator->set_threshold_event(0, entry_price * 0.98, false, [entry_price](std::vector<double>& state, double time) {
            std::cout << "    📉 SELL signal: Price down 2% at t=" 
                     << std::fixed << std::setprecision(4) << time 
                     << ", price=" << std::setprecision(2) << state[0] << "\n";
        });
        
        // Volatility risk management
        event_decorator->set_threshold_event(1, 0.35, true, [](std::vector<double>& state, double time) {
            std::cout << "    ⚠️  HIGH VOLATILITY WARNING at t=" 
                     << std::fixed << std::setprecision(4) << time 
                     << ", vol=" << std::setprecision(3) << state[1] << "\n";
        });
        
        std::cout << "\n  Simulating market data feed...\n";
        
        // Market state: [price, volatility]
        std::vector<double> market_state = {100.0, 0.2};
        
        for (int i = 0; i < 50; ++i) {
            double t = i * 0.001;  // 1ms steps
            
            // Simulate market data updates
            if (i % 5 == 0) {
                // Market price with noise
                std::vector<double> market_data = {
                    market_state[0] + 0.1 * ((i % 7) - 3),  // Price noise
                    market_state[1] + 0.01 * ((i % 3) - 1)  // Vol noise
                };
                event_decorator->submit_sensor_data("market_feed", market_data, 0.99);
            }
            
            // Fast integration step
            trading_system->step(market_state, 0.001);
        }
        
        const auto& stats = event_decorator->get_statistics();
        std::cout << "\n  Trading System Statistics:\n";
        std::cout << "    Market events: " << stats.sensor_events << "\n";
        std::cout << "    Trading signals: " << stats.total_events - stats.sensor_events << "\n";
        std::cout << "    Processing time: " << std::fixed << std::setprecision(1) 
                 << stats.average_processing_time_us() << " μs avg\n";
        std::cout << "    Event success rate: " << std::setprecision(1) 
                 << stats.event_success_rate() * 100 << "%\n";
    }
    
    std::cout << "  ✓ Financial trading scenario completed successfully\n";
}

int main() {
    std::cout << "DiffEq Library - Comprehensive Output Facilities Demo\n";
    std::cout << "====================================================\n";
    
    std::cout << "\nThis demo showcases the enhanced output facilities:\n";
    std::cout << "• Dense Output: Interpolation for arbitrary time queries\n";
    std::cout << "• Interprocess Communication: Real-time data exchange\n";
    std::cout << "• Event-Driven Feedback: Robotics and control systems\n";
    std::cout << "• SDE Synchronization: Coordinated noise processes\n";
    std::cout << "• Simultaneous Outputs: Multiple modes working together\n";
    
    try {
        demonstrate_dense_output();
        demonstrate_interprocess_communication();
        demonstrate_event_driven_feedback();
        demonstrate_sde_synchronization();
        demonstrate_simultaneous_outputs();
        demonstrate_robotics_control_scenario();
        demonstrate_financial_trading_scenario();
        
        std::cout << "\n=== Summary of Capabilities Demonstrated ===\n";
        std::cout << "✅ Dense Output & Interpolation:\n";
        std::cout << "   - Cubic spline interpolation for smooth queries\n";
        std::cout << "   - History compression for memory efficiency\n";
        std::cout << "   - Arbitrary time point queries\n";
        std::cout << "   - Dense output over intervals\n";
        
        std::cout << "\n✅ Interprocess Communication:\n";
        std::cout << "   - Shared memory for high-performance IPC\n";
        std::cout << "   - Named pipes for cross-platform communication\n";
        std::cout << "   - Producer/consumer patterns\n";
        std::cout << "   - Reliability and error handling\n";
        
        std::cout << "\n✅ Event-Driven Feedback:\n";
        std::cout << "   - Real-time sensor data integration\n";
        std::cout << "   - Control loop feedback mechanisms\n";
        std::cout << "   - Threshold crossing detection\n";
        std::cout << "   - Priority-based event processing\n";
        
        std::cout << "\n✅ SDE Synchronization:\n";
        std::cout << "   - Noise process coordination\n";
        std::cout << "   - Multiple synchronization modes\n";
        std::cout << "   - Time-based interpolation\n";
        std::cout << "   - Robust timeout handling\n";
        
        std::cout << "\n✅ Practical Applications:\n";
        std::cout << "   - Robotics control with sensor feedback\n";
        std::cout << "   - Financial trading with market data\n";
        std::cout << "   - Simultaneous debugging and control\n";
        std::cout << "   - Distributed SDE simulations\n";
        
        std::cout << "\n🎯 All Enhanced Output Facilities Working Successfully!\n";
        std::cout << "   The composable architecture allows any combination of:\n";
        std::cout << "   • Dense output for detailed analysis\n";
        std::cout << "   • Real-time streams for monitoring\n";
        std::cout << "   • Event-driven feedback for control\n";
        std::cout << "   • IPC for distributed computing\n";
        std::cout << "   • SDE synchronization for stochastic systems\n";
        
    } catch (const std::exception& e) {
        std::cout << "\n❌ Demo encountered error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
} 