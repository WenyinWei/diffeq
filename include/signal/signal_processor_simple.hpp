#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <any>

// Forward declaration
namespace diffeq::async {
    template<typename S>
    class AsyncIntegrator;
}

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
    Signal(U&& d, std::string_view id = "", double prio = 1.0) 
        : data(std::forward<U>(d))
        , timestamp(std::chrono::steady_clock::now())
        , type_id(id)
        , priority(prio) {}
};

/**
 * @brief Lightweight signal processor using standard C++ only
 * 
 * This replaces the complex event bus system with a simple, 
 * type-safe signal handling mechanism.
 */
template<typename S>
class SignalProcessor {
public:
    using state_type = S;
    using time_type = typename S::value_type;
    
    // Signal handler function types
    template<typename T>
    using SignalHandler = std::function<void(const Signal<T>&)>;
    
private:
    // Generic signal handlers using type erasure
    std::unordered_map<std::string, std::function<void(const std::any&)>> custom_handlers_;
    
    // Parameter update handlers
    std::unordered_map<std::string, std::function<void(double)>> parameter_handlers_;
    
public:
    SignalProcessor() = default;
    
    /**
     * @brief Emit a signal with arbitrary data
     */
    template<typename T>
    void emit_signal(std::string_view signal_type, T&& data, double priority = 1.0) {
        Signal<std::decay_t<T>> signal(std::forward<T>(data), signal_type, priority);
        process_signal(signal);
    }
    
    /**
     * @brief Register a typed signal handler
     */
    template<typename T, typename Handler>
    void register_handler(std::string_view signal_type, Handler&& handler) {
        static_assert(std::is_invocable_v<Handler, const Signal<T>&>, 
                      "Handler must be callable with Signal<T>");
        
        custom_handlers_[std::string(signal_type)] = 
            [h = std::forward<Handler>(handler)](const std::any& signal) {
                try {
                    const auto& typed_signal = std::any_cast<const Signal<T>&>(signal);
                    h(typed_signal);
                } catch (const std::bad_any_cast&) {
                    // Type mismatch - ignore or log
                }
            };
    }
    
    /**
     * @brief Update integration parameters
     */
    void update_parameter(const std::string& param_name, double value) {
        auto it = parameter_handlers_.find(param_name);
        if (it != parameter_handlers_.end()) {
            it->second(value);
        }
    }
    
private:
    /**
     * @brief Process a signal through registered handlers
     */
    template<typename SignalType>
    void process_signal(const SignalType& signal) {
        // Find and call appropriate handler
        auto it = custom_handlers_.find(signal.type_id);
        if (it != custom_handlers_.end()) {
            it->second(std::any(signal));
        }
    }
};

/**
 * @brief Factory function to create a signal processor
 */
template<typename S>
auto make_signal_processor(std::shared_ptr<async::AsyncIntegrator<S>> integrator = nullptr) {
    return std::make_shared<SignalProcessor<S>>();
}

/**
 * @brief Convenience overload without integrator
 */
template<typename S>
auto make_signal_processor() {
    return std::make_shared<SignalProcessor<S>>();
}

/**
 * @brief Common signal types for convenience
 */
namespace signals {
    using DoubleSignal = Signal<double>;
    using VectorSignal = Signal<std::vector<double>>;
    using StringSignal = Signal<std::string>;
}

} // namespace diffeq::signal
