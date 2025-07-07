#pragma once

#include <core/concepts.hpp>
#include <signal/signal_processor.hpp>
#include <functional>
#include <memory>
#include <vector>
#include <chrono>
#include <optional>
#include <any>

namespace diffeq::interfaces {

/**
 * @brief General interface for ODE integration with real-time signal processing
 * 
 * This unified interface handles all integration scenarios:
 * 1. Signal-triggered state modifications (discrete events)
 * 2. Signal-induced trajectory shifts (continuous influence)
 * 3. Real-time output streaming during integration
 * 4. Bidirectional communication between ODE and external processes
 */
template<system_state StateType, can_be_time TimeType>
class IntegrationInterface {
public:
    using state_type = StateType;
    using time_type = TimeType;
    using signal_processor_type = signal::SignalProcessor<StateType>;
    
    /**
     * @brief Signal influence modes
     */
    enum class InfluenceMode {
        DISCRETE_EVENT,     // Signal causes instantaneous state jump
        CONTINUOUS_SHIFT,   // Signal modifies ODE trajectory continuously  
        PARAMETER_UPDATE,   // Signal updates integration parameters
        OUTPUT_TRIGGER      // Signal triggers output/logging
    };
    
    /**
     * @brief Signal influence descriptor
     */
    struct SignalInfluence {
        InfluenceMode mode;
        std::string signal_type;
        std::function<void(const std::any&, state_type&, time_type)> handler;
        double priority = 1.0;
        bool is_active = true;
    };
    
    /**
     * @brief Output stream descriptor
     */
    struct OutputStream {
        std::string stream_id;
        std::function<void(const state_type&, time_type)> output_func;
        std::chrono::microseconds interval{1000}; // Output frequency
        std::chrono::steady_clock::time_point last_output;
        bool is_active = true;
    };
    
private:
    std::shared_ptr<signal_processor_type> signal_processor_;
    std::vector<SignalInfluence> signal_influences_;
    std::vector<OutputStream> output_streams_;
    
    // Current integration state
    std::optional<state_type> current_state_;
    time_type current_time_ = {};
    
    // Trajectory modification functions
    std::vector<std::function<void(time_type, state_type&, state_type&)>> trajectory_modifiers_;
    
public:
    explicit IntegrationInterface(std::shared_ptr<signal_processor_type> processor = nullptr)
        : signal_processor_(processor ? processor : signal::make_signal_processor<StateType>()) {
        setup_signal_handling();
    }
    
    /**
     * @brief Register a signal influence on the ODE system
     */
    template<typename SignalDataType>
    void register_signal_influence(
        std::string_view signal_type,
        InfluenceMode mode,
        std::function<void(const SignalDataType&, state_type&, time_type)> handler,
        double priority = 1.0) {
        
        SignalInfluence influence{
            .mode = mode,
            .signal_type = std::string(signal_type),
            .handler = [h = std::move(handler)](const std::any& signal_data, state_type& state, time_type t) {
                try {
                    const auto& typed_data = std::any_cast<const SignalDataType&>(signal_data);
                    h(typed_data, state, t);
                } catch (const std::bad_any_cast&) {
                    // Type mismatch - ignore silently
                }
            },
            .priority = priority,
            .is_active = true
        };
        
        signal_influences_.push_back(std::move(influence));
        
        // Register with signal processor
        signal_processor_->template register_handler<SignalDataType>(signal_type,
            [this, signal_type_str = std::string(signal_type)](const signal::Signal<SignalDataType>& sig) {
                handle_signal(signal_type_str, sig.data);
            });
    }
    
    /**
     * @brief Register an output stream for real-time data export
     */
    void register_output_stream(
        std::string_view stream_id,
        std::function<void(const state_type&, time_type)> output_func,
        std::chrono::microseconds interval = std::chrono::microseconds{1000}) {
        
        OutputStream stream{
            .stream_id = std::string(stream_id),
            .output_func = std::move(output_func),
            .interval = interval,
            .last_output = std::chrono::steady_clock::now(),
            .is_active = true
        };
        
        output_streams_.push_back(std::move(stream));
    }
    
    /**
     * @brief Add a continuous trajectory modifier
     * 
     * These functions are called during ODE evaluation to modify the dynamics
     * based on accumulated signal influences.
     */
    void add_trajectory_modifier(
        std::function<void(time_type, state_type&, state_type&)> modifier) {
        trajectory_modifiers_.push_back(std::move(modifier));
    }
    
    /**
     * @brief ODE system wrapper that incorporates signal influences
     * 
     * This function should be passed to your integrator. It wraps your
     * original ODE system and adds signal-based modifications.
     */
    template<typename OriginalODE>
    auto make_signal_aware_ode(OriginalODE&& original_ode) {
        return [this, ode = std::forward<OriginalODE>(original_ode)]
               (time_type t, const state_type& y, state_type& dydt) {
            
            // Update current state for signal handling
            current_state_ = y;
            current_time_ = t;
            
            // Compute original dynamics
            ode(t, y, dydt);
            
            // Apply trajectory modifiers from continuous signal influences
            state_type modified_state = y;
            for (auto& modifier : trajectory_modifiers_) {
                modifier(t, modified_state, dydt);
            }
            
            // Handle real-time outputs
            process_output_streams(y, t);
        };
    }
    
    /**
     * @brief Process discrete events (instantaneous state modifications)
     */
    void apply_discrete_event(const std::string& signal_type, const std::any& signal_data) {
        if (!current_state_) return;
        
        for (auto& influence : signal_influences_) {
            if (influence.signal_type == signal_type && 
                influence.mode == InfluenceMode::DISCRETE_EVENT &&
                influence.is_active) {
                
                influence.handler(signal_data, *current_state_, current_time_);
            }
        }
    }
    
    /**
     * @brief Get current integration state
     */
    std::optional<state_type> get_current_state() const {
        return current_state_;
    }
    
    /**
     * @brief Get current time
     */
    time_type get_current_time() const {
        return current_time_;
    }
    
    /**
     * @brief Enable/disable signal influence
     */
    void set_signal_influence_active(const std::string& signal_type, bool active) {
        for (auto& influence : signal_influences_) {
            if (influence.signal_type == signal_type) {
                influence.is_active = active;
            }
        }
    }
    
    /**
     * @brief Enable/disable output stream
     */
    void set_output_stream_active(const std::string& stream_id, bool active) {
        for (auto& stream : output_streams_) {
            if (stream.stream_id == stream_id) {
                stream.is_active = active;
            }
        }
    }
    
    /**
     * @brief Get signal processor for direct access
     */
    std::shared_ptr<signal_processor_type> get_signal_processor() {
        return signal_processor_;
    }
    
private:
    void setup_signal_handling() {
        // Basic signal handling is set up when influences are registered
    }
    
    void handle_signal(const std::string& signal_type, const std::any& signal_data) {
        // Handle different influence modes
        for (auto& influence : signal_influences_) {
            if (influence.signal_type == signal_type && influence.is_active) {
                
                if (influence.mode == InfluenceMode::DISCRETE_EVENT && current_state_) {
                    // Apply immediate state modification
                    influence.handler(signal_data, *current_state_, current_time_);
                    
                } else if (influence.mode == InfluenceMode::CONTINUOUS_SHIFT) {
                    // Add to trajectory modifiers for continuous influence
                    add_trajectory_modifier([influence, signal_data]
                        (time_type t, state_type& state, state_type& /* dydt */) {
                        influence.handler(signal_data, state, t);
                    });
                    
                } else if (influence.mode == InfluenceMode::PARAMETER_UPDATE) {
                    // Handle parameter updates (integrator-specific)
                    if (current_state_) {
                        influence.handler(signal_data, *current_state_, current_time_);
                    }
                    
                } else if (influence.mode == InfluenceMode::OUTPUT_TRIGGER && current_state_) {
                    // Trigger immediate output
                    for (auto& stream : output_streams_) {
                        if (stream.is_active) {
                            stream.output_func(*current_state_, current_time_);
                        }
                    }
                }
            }
        }
    }
    
    void process_output_streams(const state_type& state, time_type t) {
        auto now = std::chrono::steady_clock::now();
        
        for (auto& stream : output_streams_) {
            if (!stream.is_active) continue;
            
            if (now - stream.last_output >= stream.interval) {
                stream.output_func(state, t);
                stream.last_output = now;
            }
        }
    }
};

/**
 * @brief Factory function to create an integration interface
 */
template<system_state StateType, can_be_time TimeType = double>
auto make_integration_interface(
    std::shared_ptr<signal::SignalProcessor<StateType>> processor) {
    return std::make_unique<IntegrationInterface<StateType, TimeType>>(processor);
}

/**
 * @brief Convenience factory with no processor argument
 */
template<system_state StateType, can_be_time TimeType = double>
auto make_integration_interface() {
    return std::make_unique<IntegrationInterface<StateType, TimeType>>();
}

} // namespace diffeq::interfaces
