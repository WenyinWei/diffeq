#pragma once

#include "integrator_decorator.hpp"
#include <chrono>
#include <future>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace diffeq::core::composable {

/**
 * @brief Timeout configuration for integration protection
 */
struct TimeoutConfig {
    std::chrono::milliseconds timeout_duration{5000};
    bool throw_on_timeout{true};
    bool enable_progress_callback{false};
    std::chrono::milliseconds progress_interval{100};
    std::function<bool(double, double, std::chrono::milliseconds)> progress_callback;
    
    // Validation settings
    bool validate_timeout_duration{true};
    std::chrono::milliseconds min_timeout_duration{10};  // Minimum 10ms
    std::chrono::milliseconds max_timeout_duration{std::chrono::hours{24}};  // Maximum 24h
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (validate_timeout_duration) {
            if (timeout_duration < min_timeout_duration) {
                throw std::invalid_argument(
                    "Timeout duration " + std::to_string(timeout_duration.count()) + 
                    "ms is below minimum " + std::to_string(min_timeout_duration.count()) + "ms");
            }
            if (timeout_duration > max_timeout_duration) {
                throw std::invalid_argument(
                    "Timeout duration " + std::to_string(timeout_duration.count()) + 
                    "ms exceeds maximum " + std::to_string(max_timeout_duration.count()) + "ms");
            }
        }
        
        if (enable_progress_callback && !progress_callback) {
            throw std::invalid_argument("Progress callback enabled but no callback provided");
        }
        
        if (progress_interval <= std::chrono::milliseconds{0}) {
            throw std::invalid_argument("Progress interval must be positive");
        }
    }
};

/**
 * @brief Result information for timeout-protected integration
 */
struct TimeoutResult {
    bool completed{false};
    std::chrono::milliseconds elapsed_time{0};
    double final_time{0.0};
    std::string error_message;
    bool user_cancelled{false};
    size_t progress_callbacks_made{0};
    
    bool is_success() const { return completed && error_message.empty() && !user_cancelled; }
    bool is_timeout() const { return !completed && error_message.find("timeout") != std::string::npos; }
    bool is_user_cancelled() const { return user_cancelled; }
    bool is_error() const { return !completed && !error_message.empty() && !is_timeout(); }
    
    /**
     * @brief Get human-readable status description
     */
    std::string status_description() const {
        if (is_success()) return "Integration completed successfully";
        if (is_timeout()) return "Integration timed out";
        if (is_user_cancelled()) return "Integration cancelled by user";
        if (is_error()) return "Integration failed: " + error_message;
        return "Integration status unknown";
    }
};

/**
 * @brief Timeout decorator - adds timeout protection to any integrator
 * 
 * This decorator provides robust timeout protection with the following features:
 * - Configurable timeout duration with validation
 * - Optional progress monitoring with user cancellation
 * - Comprehensive error handling and reporting
 * - Thread-safe async execution with proper cleanup
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles timeout protection
 * - No Dependencies: Works with any integrator type
 * - Robust Error Handling: Graceful failure modes
 * - User Control: Progress callbacks and cancellation support
 */
template<system_state S, can_be_time T = double>
class TimeoutDecorator : public IntegratorDecorator<S, T> {
private:
    TimeoutConfig config_;

public:
    /**
     * @brief Construct timeout decorator
     * @param integrator The integrator to wrap
     * @param config Timeout configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit TimeoutDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator, 
                             TimeoutConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {
        config_.validate();
    }

    /**
     * @brief Main timeout-protected integration method
     * @param state Initial state (modified in-place)
     * @param dt Time step
     * @param end_time Final integration time
     * @return Detailed result with timing and status information
     */
    TimeoutResult integrate_with_timeout(typename IntegratorDecorator<S, T>::state_type& state, 
                                        T dt, T end_time) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        TimeoutResult result;
        result.final_time = this->current_time();

        try {
            if (config_.enable_progress_callback && config_.progress_callback) {
                result = integrate_with_progress_monitoring(state, dt, end_time, start_time);
            } else {
                result = integrate_with_simple_timeout(state, dt, end_time, start_time);
            }
            
        } catch (const std::exception& e) {
            result.completed = false;
            result.error_message = "Integration failed: " + std::string(e.what());
        }

        const auto end_time_clock = std::chrono::high_resolution_clock::now();
        result.elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_clock - start_time);

        // Handle timeout according to configuration
        if (!result.completed && config_.throw_on_timeout && result.is_timeout()) {
            throw std::runtime_error(result.error_message);
        }

        return result;
    }

    /**
     * @brief Override standard integrate to use timeout protection
     */
    void integrate(typename IntegratorDecorator<S, T>::state_type& state, T dt, T end_time) override {
        auto result = integrate_with_timeout(state, dt, end_time);
        if (!result.is_success() && config_.throw_on_timeout) {
            throw std::runtime_error(result.status_description());
        }
    }

    /**
     * @brief Access and modify timeout configuration
     */
    TimeoutConfig& config() { return config_; }
    const TimeoutConfig& config() const { return config_; }
    
    /**
     * @brief Update timeout configuration with validation
     * @param new_config New configuration
     * @throws std::invalid_argument if new config is invalid
     */
    void update_config(TimeoutConfig new_config) {
        new_config.validate();
        config_ = std::move(new_config);
    }

private:
    /**
     * @brief Simple timeout without progress monitoring
     */
    TimeoutResult integrate_with_simple_timeout(
        typename IntegratorDecorator<S, T>::state_type& state,
        T dt, T end_time,
        std::chrono::high_resolution_clock::time_point start_time) {
        
        TimeoutResult result;
        
        auto future = std::async(std::launch::async, [this, &state, dt, end_time]() {
            this->wrapped_integrator_->integrate(state, dt, end_time);
        });

        if (future.wait_for(config_.timeout_duration) == std::future_status::timeout) {
            result.completed = false;
            result.error_message = "Integration timed out after " + 
                                 std::to_string(config_.timeout_duration.count()) + "ms";
        } else {
            future.get();  // May throw if integration failed
            result.completed = true;
            result.final_time = this->current_time();
        }
        
        return result;
    }

    /**
     * @brief Integration with progress monitoring and user cancellation
     */
    TimeoutResult integrate_with_progress_monitoring(
        typename IntegratorDecorator<S, T>::state_type& state,
        T dt, T end_time,
        std::chrono::high_resolution_clock::time_point start_time) {
        
        TimeoutResult result;
        
        auto future = std::async(std::launch::async, [this, &state, dt, end_time]() {
            this->wrapped_integrator_->integrate(state, dt, end_time);
        });

        auto last_progress_check = start_time;
        auto timeout_deadline = start_time + config_.timeout_duration;
        
        // Monitor progress until completion or timeout
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
            
            // Check if integration completed
            if (future.wait_for(std::chrono::milliseconds{1}) == std::future_status::ready) {
                future.get();  // May throw if integration failed
                result.completed = true;
                result.final_time = this->current_time();
                break;
            }
            
            // Check for timeout
            if (now >= timeout_deadline) {
                result.completed = false;
                result.error_message = "Integration timed out after " + 
                                     std::to_string(config_.timeout_duration.count()) + "ms";
                break;
            }
            
            // Check if time for progress callback
            if (now - last_progress_check >= config_.progress_interval) {
                double current_time = this->current_time();
                bool should_continue = config_.progress_callback(
                    current_time, static_cast<double>(end_time), elapsed);
                
                result.progress_callbacks_made++;
                last_progress_check = now;
                
                // Check for user cancellation
                if (!should_continue) {
                    result.completed = false;
                    result.user_cancelled = true;
                    result.error_message = "Integration cancelled by user";
                    break;
                }
            }
            
            // Brief sleep to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
        
        return result;
    }
};

} // namespace diffeq::core::composable