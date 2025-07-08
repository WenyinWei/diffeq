#pragma once

#include <future>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <functional>
#include "concepts.hpp"
#include "abstract_integrator.hpp"

namespace diffeq::core {

/**
 * @brief Exception thrown when integration times out
 */
class IntegrationTimeoutException : public std::runtime_error {
public:
    explicit IntegrationTimeoutException(const std::string& message)
        : std::runtime_error(message) {}
    
    explicit IntegrationTimeoutException(std::chrono::milliseconds timeout_duration)
        : std::runtime_error("Integration timed out after " + 
                           std::to_string(timeout_duration.count()) + "ms") {}
};

/**
 * @brief Configuration for timeout-enabled integration
 */
struct TimeoutConfig {
    std::chrono::milliseconds timeout_duration{5000};  // Default 5 seconds
    bool throw_on_timeout{true};                        // Throw exception vs return false
    bool enable_progress_callback{false};               // Enable progress monitoring
    std::chrono::milliseconds progress_interval{100};   // Progress callback interval
    
    // Optional progress callback: (current_time, end_time, elapsed_time) -> should_continue
    std::function<bool(double, double, std::chrono::milliseconds)> progress_callback;
};

/**
 * @brief Result of a timeout-enabled integration
 */
struct IntegrationResult {
    bool completed{false};                          // Whether integration completed successfully
    std::chrono::milliseconds elapsed_time{0};     // Total elapsed time
    double final_time{0.0};                       // Final integration time reached
    std::string error_message;                     // Error message if failed
    
    // Success indicators
    bool is_success() const { return completed && error_message.empty(); }
    bool is_timeout() const { return !completed && error_message.find("timeout") != std::string::npos; }
    bool is_error() const { return !completed && !is_timeout(); }
};

/**
 * @brief Timeout-enabled integration wrapper
 * 
 * This class provides timeout protection for any integrator by wrapping
 * the integration call in an async operation with configurable timeout.
 * 
 * Features:
 * - Configurable timeout duration
 * - Progress monitoring with callbacks
 * - Detailed result information
 * - Exception vs return value error handling
 * - Compatible with all integrator types
 * 
 * @tparam Integrator The integrator type to wrap
 */
template<typename Integrator>
class TimeoutIntegrator {
public:
    using integrator_type = Integrator;
    using state_type = typename Integrator::state_type;
    using time_type = typename Integrator::time_type;

    /**
     * @brief Construct timeout integrator with an existing integrator
     * @param integrator The integrator to wrap (moved)
     * @param config Timeout configuration
     */
    explicit TimeoutIntegrator(Integrator integrator, TimeoutConfig config = {})
        : integrator_(std::move(integrator)), config_(std::move(config)) {}

    /**
     * @brief Construct timeout integrator with integrator parameters
     * @param config Timeout configuration
     * @param args Arguments forwarded to integrator constructor
     */
    template<typename... Args>
    explicit TimeoutIntegrator(TimeoutConfig config, Args&&... args)
        : integrator_(std::forward<Args>(args)...), config_(std::move(config)) {}

    /**
     * @brief Perform timeout-protected integration
     * @param state State vector to integrate (modified in-place)
     * @param dt Time step size
     * @param end_time Final integration time
     * @return Integration result with timing and success information
     */
    IntegrationResult integrate_with_timeout(state_type& state, time_type dt, time_type end_time) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        IntegrationResult result;
        result.final_time = integrator_.current_time();

        try {
            // Launch integration in async task
            auto future = std::async(std::launch::async, [this, &state, dt, end_time]() {
                integrator_.integrate(state, dt, end_time);
            });

            // Wait with timeout
            if (config_.enable_progress_callback && config_.progress_callback) {
                // Monitor progress with callbacks
                result = wait_with_progress_monitoring(future, end_time, start_time);
            } else {
                // Simple timeout wait
                if (future.wait_for(config_.timeout_duration) == std::future_status::timeout) {
                    result.completed = false;
                    result.error_message = "Integration timed out after " + 
                                         std::to_string(config_.timeout_duration.count()) + "ms";
                } else {
                    future.get(); // Check for exceptions
                    result.completed = true;
                }
            }
            
            result.final_time = integrator_.current_time();
            
        } catch (const std::exception& e) {
            result.completed = false;
            result.error_message = "Integration failed: " + std::string(e.what());
        }

        // Calculate elapsed time
        const auto end_clock_time = std::chrono::high_resolution_clock::now();
        result.elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_clock_time - start_time);

        // Handle timeout according to configuration
        if (!result.completed && config_.throw_on_timeout && result.is_timeout()) {
            throw IntegrationTimeoutException(result.elapsed_time);
        }

        return result;
    }

    /**
     * @brief Simple timeout integration (backwards compatibility)
     * @param state State vector to integrate
     * @param dt Time step size  
     * @param end_time Final integration time
     * @param timeout Optional timeout override
     * @return true if completed successfully, false if timed out
     */
    bool integrate_with_timeout_simple(state_type& state, time_type dt, time_type end_time,
                                      std::chrono::milliseconds timeout = std::chrono::milliseconds{0}) {
        auto old_config = config_;
        if (timeout.count() > 0) {
            config_.timeout_duration = timeout;
        }
        config_.throw_on_timeout = false;
        
        auto result = integrate_with_timeout(state, dt, end_time);
        config_ = old_config; // Restore original config
        
        return result.completed;
    }

    /**
     * @brief Access the underlying integrator
     * @return Reference to the wrapped integrator
     */
    Integrator& integrator() { return integrator_; }
    const Integrator& integrator() const { return integrator_; }

    /**
     * @brief Access timeout configuration
     * @return Reference to the timeout configuration
     */
    TimeoutConfig& config() { return config_; }
    const TimeoutConfig& config() const { return config_; }

private:
    Integrator integrator_;
    TimeoutConfig config_;

    /**
     * @brief Wait for future completion with progress monitoring
     */
    IntegrationResult wait_with_progress_monitoring(std::future<void>& future, time_type end_time,
                                                   std::chrono::high_resolution_clock::time_point start_time) {
        IntegrationResult result;
        
        auto last_progress_time = start_time;
        
        while (true) {
            // Check if integration completed
            if (future.wait_for(config_.progress_interval) == std::future_status::ready) {
                future.get(); // Check for exceptions
                result.completed = true;
                break;
            }
            
            // Check total timeout
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            
            if (elapsed >= config_.timeout_duration) {
                result.completed = false;
                result.error_message = "Integration timed out after " + 
                                     std::to_string(elapsed.count()) + "ms";
                break;
            }
            
            // Call progress callback
            if (config_.progress_callback) {
                bool should_continue = config_.progress_callback(
                    static_cast<double>(integrator_.current_time()),
                    static_cast<double>(end_time), 
                    elapsed
                );
                
                if (!should_continue) {
                    result.completed = false;
                    result.error_message = "Integration cancelled by progress callback";
                    break;
                }
            }
            
            last_progress_time = current_time;
        }
        
        return result;
    }
};

/**
 * @brief Factory function to create timeout-enabled integrator
 * @tparam Integrator The integrator type
 * @param integrator The integrator to wrap
 * @param config Timeout configuration
 * @return TimeoutIntegrator wrapping the provided integrator
 */
template<typename Integrator>
auto make_timeout_integrator(Integrator integrator, TimeoutConfig config = {}) {
    return TimeoutIntegrator<Integrator>(std::move(integrator), std::move(config));
}

/**
 * @brief Convenience function for simple timeout integration
 * @tparam Integrator The integrator type
 * @param integrator The integrator to use
 * @param state State vector to integrate
 * @param dt Time step size
 * @param end_time Final integration time
 * @param timeout Timeout duration
 * @return true if completed successfully, false if timed out
 */
template<typename Integrator, typename State>
bool integrate_with_timeout(Integrator& integrator, State& state, 
                           typename Integrator::time_type dt, 
                           typename Integrator::time_type end_time,
                           std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
    // Create a copy of the state to avoid race conditions
    State state_copy = state;
    
    auto future = std::async(std::launch::async, [&integrator, &state_copy, dt, end_time]() {
        integrator.integrate(state_copy, dt, end_time);
    });
    
    auto status = future.wait_for(timeout);
    
    if (status == std::future_status::ready) {
        try {
            future.get(); // Check for exceptions
            state = state_copy; // Update original state only if successful
            return true;
        } catch (...) {
            return false;
        }
    } else {
        return false; // Timed out
    }
}

} // namespace diffeq::core