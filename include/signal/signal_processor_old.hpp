#pragma once

#include <async/async_integrator.hpp>
#include <functional>
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace diffeq::signal {

/**
 * @brief Generic signal data structure
 * 
 * This replaces the complex communication-specific message types
 * with a simple, generic signal that can carry any data type.
 */
template<typename T>
struct Signal {
    T data;
    std::chrono::steady_clock::time_point timestamp;
    std::string type_id;
    double priority = 1.0;
    
    template<typename U>
    Signal(U&& signal_data, std::string type = "", double prio = 1.0)
        : data(std::forward<U>(signal_data))
        , timestamp(std::chrono::steady_clock::now())
        , type_id(std::move(type))
        , priority(prio) {}
};

/**
 * @brief Signal handler concept
 */
template<typename Handler, typename Signal>
concept SignalHandler = requires(Handler h, const Signal& s) {
    { h(s) } -> std::same_as<void>;
};

/**
 * @brief Simple signal processor for integrator callbacks
 * 
 * This provides a lightweight way to process signals and update
 * integration parameters without heavy communication dependencies.
 */
template<system_state S, can_be_time T = double>
class SignalProcessor {
public:
    using state_type = S;
    using time_type = T;
    using value_type = typename S::value_type;
    using integrator_type = AsyncIntegrator<S, T>;
    
    // Signal types for different domains
    using parameter_signal = Signal<std::unordered_map<std::string, double>>;
    using control_signal = Signal<std::vector<double>>;
    using market_signal = Signal<std::unordered_map<std::string, double>>;
    
    explicit SignalProcessor(std::shared_ptr<integrator_type> integrator)
        : integrator_(std::move(integrator)) {
        
        // Set up default callbacks
        setup_default_callbacks();
    }
    
    /**
     * @brief Process parameter update signal
     */
    void process_parameter_signal(const parameter_signal& signal) {
        for (const auto& [name, value] : signal.data) {
            update_parameter(name, value);
        }
    }
    
    /**
     * @brief Process control signal (for robotics)
     */
    void process_control_signal(const control_signal& signal) {
        // Update control targets
        if (signal.data.size() >= 6) { // 6-DOF robot example
            control_targets_ = signal.data;
            last_control_update_ = signal.timestamp;
        }
    }
    
    /**
     * @brief Process market signal (for finance)
     */
    void process_market_signal(const market_signal& signal) {
        // Update market data
        for (const auto& [symbol, value] : signal.data) {
            market_data_[symbol] = value;
        }
        last_market_update_ = signal.timestamp;
    }
    
    /**
     * @brief Get current control targets
     */
    const std::vector<double>& get_control_targets() const {
        return control_targets_;
    }
    
    /**
     * @brief Get market data for symbol
     */
    double get_market_data(const std::string& symbol) const {
        auto it = market_data_.find(symbol);
        return (it != market_data_.end()) ? it->second : 0.0;
    }
    
    /**
     * @brief Update integration parameter
     */
    void update_parameter(const std::string& name, double value) {
        parameters_[name] = value;
        
        // Notify integrator asynchronously
        if (integrator_) {
            integrator_->update_parameter_async(name, value);
        }
    }
    
    /**
     * @brief Get parameter value
     */
    double get_parameter(const std::string& name) const {
        auto it = parameters_.find(name);
        return (it != parameters_.end()) ? it->second : 0.0;
    }
    
    /**
     * @brief Register custom signal handler
     */
    template<typename SignalType, SignalHandler<SignalType> Handler>
    void register_handler(const std::string& signal_type, Handler&& handler) {
        // Store handler in type-erased form
        custom_handlers_[signal_type] = [h = std::forward<Handler>(handler)](const std::any& signal) {
            try {
                const auto& typed_signal = std::any_cast<const SignalType&>(signal);
                h(typed_signal);
            } catch (const std::bad_any_cast&) {
                // Handle type mismatch gracefully
            }
        };
    }
    
    /**
     * @brief Process generic signal
     */
    template<typename SignalType>
    void process_signal(const SignalType& signal) {
        auto it = custom_handlers_.find(signal.type_id);
        if (it != custom_handlers_.end()) {
            it->second(std::any(signal));
        }
    }

private:
    std::shared_ptr<integrator_type> integrator_;
    
    // Signal data storage
    std::unordered_map<std::string, double> parameters_;
    std::vector<double> control_targets_;
    std::unordered_map<std::string, double> market_data_;
    
    // Timestamps
    std::chrono::steady_clock::time_point last_control_update_;
    std::chrono::steady_clock::time_point last_market_update_;
    
    // Custom handlers
    std::unordered_map<std::string, std::function<void(const std::any&)>> custom_handlers_;
    
    void setup_default_callbacks() {
        if (!integrator_) return;
        
        // Set up step callback to update internal state
        integrator_->set_step_callback([this](const state_type& state, time_type t) {
            on_step_completed(state, t);
        });
        
        // Set up parameter callback
        integrator_->set_parameter_callback([this](const std::string& name, double value) {
            on_parameter_updated(name, value);
        });
        
        // Set up emergency callback
        integrator_->set_emergency_callback([this]() {
            on_emergency_stop();
        });
    }
    
    void on_step_completed(const state_type& state, time_type t) {
        // Override in derived classes for domain-specific processing
    }
    
    void on_parameter_updated(const std::string& name, double value) {
        // Handle parameter updates
        parameters_[name] = value;
    }
    
    void on_emergency_stop() {
        // Clear all targets and data
        control_targets_.clear();
        // Additional emergency procedures can be added here
    }
};

/**
 * @brief Factory function for creating signal processors
 */
template<system_state S, can_be_time T = double>
auto make_signal_processor(std::shared_ptr<AsyncIntegrator<S, T>> integrator) {
    return std::make_unique<SignalProcessor<S, T>>(std::move(integrator));
}

/**
 * @brief Convenience functions for common signal patterns
 */
template<typename SignalType>
auto make_signal(const typename SignalType::value_type& data, 
                const std::string& type = "", 
                double priority = 1.0) {
    return SignalType{data, type, priority};
}

/**
 * @brief Timer-based signal generator for testing
 */
template<typename SignalType>
class SignalGenerator {
public:
    using signal_type = SignalType;
    using generator_function = std::function<SignalType()>;
    
    SignalGenerator(generator_function gen, std::chrono::microseconds interval)
        : generator_(std::move(gen))
        , interval_(interval)
        , running_(false) {}
    
    ~SignalGenerator() {
        stop();
    }
    
    template<typename Handler>
    void start(Handler&& handler) {
        if (running_.exchange(true)) {
            return;
        }
        
        thread_ = std::thread([this, h = std::forward<Handler>(handler)]() {
            while (running_.load()) {
                auto signal = generator_();
                h(signal);
                
                std::this_thread::sleep_for(interval_);
            }
        });
    }
    
    void stop() {
        if (!running_.exchange(false)) {
            return;
        }
        
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    generator_function generator_;
    std::chrono::microseconds interval_;
    std::atomic<bool> running_;
    std::thread thread_;
};

} // namespace diffeq::signal
