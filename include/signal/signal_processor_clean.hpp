#pragma once

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <chrono>
#include <any>

namespace diffeq::signal {

/**
 * @brief Generic signal data structure
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
 * @brief Simple signal processor
 */
template<typename S>
class SignalProcessor {
public:
    using state_type = S;
    
    // Signal handler function types
    template<typename T>
    using SignalHandler = std::function<void(const Signal<T>&)>;
    
private:
    // Generic signal handlers using type erasure
    std::unordered_map<std::string, std::function<void(const std::any&)>> custom_handlers_;
    
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
        custom_handlers_[std::string(signal_type)] = 
            [h = std::forward<Handler>(handler)](const std::any& signal) {
                try {
                    const auto& typed_signal = std::any_cast<const Signal<T>&>(signal);
                    h(typed_signal);
                } catch (const std::bad_any_cast&) {
                    // Type mismatch - ignore
                }
            };
    }
    
private:
    /**
     * @brief Process a signal through registered handlers
     */
    template<typename SignalType>
    void process_signal(const SignalType& signal) {
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
std::shared_ptr<SignalProcessor<S>> make_signal_processor() {
    return std::make_shared<SignalProcessor<S>>();
}

} // namespace diffeq::signal
