#pragma once

#include "integrator_decorator.hpp"
#include <future>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>

namespace diffeq::core::composable {

/**
 * @brief Configuration for async execution
 */
struct AsyncConfig {
    size_t thread_pool_size{0};  // 0 = auto-detect
    bool enable_progress_monitoring{false};
    std::chrono::microseconds monitoring_interval{1000};
    bool enable_cancellation{true};
    std::chrono::milliseconds max_async_wait{std::chrono::minutes{10}};
    
    // Validation settings
    bool validate_thread_pool_size{true};
    size_t min_thread_pool_size{1};
    size_t max_thread_pool_size{std::thread::hardware_concurrency() * 2};
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (validate_thread_pool_size && thread_pool_size > 0) {
            if (thread_pool_size < min_thread_pool_size) {
                throw std::invalid_argument("thread_pool_size must be >= " + 
                                          std::to_string(min_thread_pool_size));
            }
            if (thread_pool_size > max_thread_pool_size) {
                throw std::invalid_argument("thread_pool_size exceeds system limit of " + 
                                          std::to_string(max_thread_pool_size));
            }
        }
        
        if (monitoring_interval <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("monitoring_interval must be positive");
        }
        
        if (max_async_wait <= std::chrono::milliseconds{0}) {
            throw std::invalid_argument("max_async_wait must be positive");
        }
    }
};

/**
 * @brief Result information for async operations
 */
struct AsyncResult {
    bool completed{false};
    bool cancelled{false};
    std::chrono::milliseconds execution_time{0};
    std::string error_message;
    size_t monitoring_checks{0};
    
    bool is_success() const { return completed && !cancelled && error_message.empty(); }
    bool is_cancelled() const { return cancelled; }
    bool is_error() const { return !completed && !cancelled && !error_message.empty(); }
    
    std::string status_description() const {
        if (is_success()) return "Async integration completed successfully";
        if (is_cancelled()) return "Async integration was cancelled";
        if (is_error()) return "Async integration failed: " + error_message;
        return "Async integration in progress";
    }
};

/**
 * @brief Async execution decorator - adds async capabilities to any integrator
 * 
 * This decorator provides asynchronous execution with the following features:
 * - Non-blocking integration and stepping
 * - Progress monitoring with cancellation support
 * - Thread-safe operations with proper resource management
 * - Configurable thread pool and monitoring intervals
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles async execution
 * - No Dependencies: Works with any integrator type
 * - Non-blocking: All operations return immediately with futures
 * - Cancellable: Support for graceful cancellation
 */
template<system_state S, can_be_time T = double>
class AsyncDecorator : public IntegratorDecorator<S, T> {
private:
    AsyncConfig config_;
    mutable std::mutex state_mutex_;
    std::atomic<bool> cancellation_requested_{false};
    std::atomic<size_t> active_operations_{0};

public:
    /**
     * @brief Construct async decorator
     * @param integrator The integrator to wrap
     * @param config Async configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit AsyncDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                           AsyncConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {
        
        config_.validate();
        
        // Auto-detect optimal thread pool size if not specified
        if (config_.thread_pool_size == 0) {
            config_.thread_pool_size = std::max(2U, std::thread::hardware_concurrency() / 2);
        }
    }

    /**
     * @brief Async integration with future return
     * @param state Initial state (must remain valid until future completes)
     * @param dt Time step
     * @param end_time Final integration time
     * @return Future that will contain the integration result
     */
    std::future<AsyncResult> integrate_async(typename IntegratorDecorator<S, T>::state_type& state, 
                                           T dt, T end_time) {
        return std::async(std::launch::async, [this, &state, dt, end_time]() -> AsyncResult {
            ++active_operations_;
            auto operation_guard = make_scope_guard([this] { --active_operations_; });
            
            const auto start_time = std::chrono::high_resolution_clock::now();
            AsyncResult result;
            
            try {
                if (config_.enable_progress_monitoring) {
                    result = integrate_with_monitoring(state, dt, end_time, start_time);
                } else {
                    result = integrate_simple(state, dt, end_time);
                }
                
                if (!cancellation_requested_.load()) {
                    result.completed = true;
                }
                
            } catch (const std::exception& e) {
                result.error_message = "Async integration failed: " + std::string(e.what());
            }
            
            const auto end_time_clock = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time_clock - start_time);
            
            return result;
        });
    }

    /**
     * @brief Async single step with future return
     * @param state Current state (must remain valid until future completes)
     * @param dt Time step
     * @return Future that will contain the step result
     */
    std::future<AsyncResult> step_async(typename IntegratorDecorator<S, T>::state_type& state, T dt) {
        return std::async(std::launch::async, [this, &state, dt]() -> AsyncResult {
            ++active_operations_;
            auto operation_guard = make_scope_guard([this] { --active_operations_; });
            
            const auto start_time = std::chrono::high_resolution_clock::now();
            AsyncResult result;
            
            try {
                if (cancellation_requested_.load()) {
                    result.cancelled = true;
                } else {
                    this->wrapped_integrator_->step(state, dt);
                    result.completed = true;
                }
                
            } catch (const std::exception& e) {
                result.error_message = "Async step failed: " + std::string(e.what());
            }
            
            const auto end_time_clock = std::chrono::high_resolution_clock::now();
            result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time_clock - start_time);
            
            return result;
        });
    }

    /**
     * @brief Request cancellation of all active async operations
     * Note: Cancellation is cooperative and may not be immediate
     */
    void request_cancellation() {
        cancellation_requested_.store(true);
    }

    /**
     * @brief Clear cancellation request
     */
    void clear_cancellation() {
        cancellation_requested_.store(false);
    }

    /**
     * @brief Check if cancellation has been requested
     */
    bool is_cancellation_requested() const {
        return cancellation_requested_.load();
    }

    /**
     * @brief Get number of currently active async operations
     */
    size_t get_active_operations_count() const {
        return active_operations_.load();
    }

    /**
     * @brief Wait for all active operations to complete
     * @param timeout Maximum time to wait
     * @return true if all operations completed within timeout
     */
    bool wait_for_all_operations(std::chrono::milliseconds timeout = std::chrono::seconds{30}) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        
        while (active_operations_.load() > 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        
        return active_operations_.load() == 0;
    }

    /**
     * @brief Access and modify async configuration
     */
    AsyncConfig& config() { return config_; }
    const AsyncConfig& config() const { return config_; }
    
    /**
     * @brief Update async configuration with validation
     * @param new_config New configuration
     * @throws std::invalid_argument if new config is invalid
     */
    void update_config(AsyncConfig new_config) {
        new_config.validate();
        std::lock_guard<std::mutex> lock(state_mutex_);
        config_ = std::move(new_config);
    }

private:
    /**
     * @brief Simple async integration without monitoring
     */
    AsyncResult integrate_simple(typename IntegratorDecorator<S, T>::state_type& state, 
                                T dt, T end_time) {
        AsyncResult result;
        
        if (cancellation_requested_.load()) {
            result.cancelled = true;
            return result;
        }
        
        this->wrapped_integrator_->integrate(state, dt, end_time);
        return result;
    }

    /**
     * @brief Async integration with progress monitoring
     */
    AsyncResult integrate_with_monitoring(typename IntegratorDecorator<S, T>::state_type& state,
                                         T dt, T end_time,
                                         std::chrono::high_resolution_clock::time_point start_time) {
        AsyncResult result;
        
        // For monitoring, we need to do step-by-step integration
        T current_time = this->current_time();
        auto last_check = start_time;
        
        while (current_time < end_time && !cancellation_requested_.load()) {
            auto now = std::chrono::high_resolution_clock::now();
            
            // Check if it's time for monitoring
            if (now - last_check >= config_.monitoring_interval) {
                result.monitoring_checks++;
                last_check = now;
            }
            
            // Perform one step
            T step_size = std::min(dt, end_time - current_time);
            this->wrapped_integrator_->step(state, step_size);
            current_time = this->current_time();
        }
        
        if (cancellation_requested_.load()) {
            result.cancelled = true;
        }
        
        return result;
    }

    /**
     * @brief RAII scope guard for operation counting
     */
    template<typename F>
    class ScopeGuard {
        F func_;
        bool active_;
    public:
        explicit ScopeGuard(F f) : func_(std::move(f)), active_(true) {}
        ~ScopeGuard() { if (active_) func_(); }
        ScopeGuard(ScopeGuard&& other) noexcept 
            : func_(std::move(other.func_)), active_(other.active_) {
            other.active_ = false;
        }
        ScopeGuard(const ScopeGuard&) = delete;
        ScopeGuard& operator=(const ScopeGuard&) = delete;
        ScopeGuard& operator=(ScopeGuard&&) = delete;
    };

    template<typename F>
    auto make_scope_guard(F&& func) {
        return ScopeGuard<std::decay_t<F>>(std::forward<F>(func));
    }
};

} // namespace diffeq::core::composable