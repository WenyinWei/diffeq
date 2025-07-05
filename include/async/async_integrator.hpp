#pragma once

#include <core/abstract_integrator.hpp>
#include <integrators/ode/rk45.hpp>
#include <integrators/ode/dop853.hpp>
#include <integrators/ode/bdf.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <functional>
#include <memory>
#include <queue>
#include <chrono>
#include <optional>
#include <type_traits>

// C++23 std::execution support with fallback
#if __has_include(<execution>) && defined(__cpp_lib_execution)
#include <execution>
#define DIFFEQ_HAS_STD_EXECUTION 1
#else
#define DIFFEQ_HAS_STD_EXECUTION 0
#endif

namespace diffeq::async {

/**
 * @brief Simple async executor using standard C++ facilities only
 * 
 * This replaces the complex communication system with a lightweight
 * standard-library-only approach suitable for C++ standard inclusion.
 */
class AsyncExecutor {
public:
    explicit AsyncExecutor(size_t num_threads = std::thread::hardware_concurrency())
        : stop_flag_(false) {
        
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] { worker_thread(); });
        }
    }
    
    ~AsyncExecutor() {
        shutdown();
    }
    
    template<typename F>
    auto submit(F&& func) -> std::future<std::invoke_result_t<F>> {
        using return_type = std::invoke_result_t<F>;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::forward<F>(func)
        );
        
        auto future = task->get_future();
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_flag_) {
                throw std::runtime_error("Executor is shutting down");
            }
            task_queue_.emplace([task] { (*task)(); });
        }
        
        condition_.notify_one();
        return future;
    }
    
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_flag_ = true;
        }
        
        condition_.notify_all();
        
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads_.clear();
    }
    
private:
    void worker_thread() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { 
                    return stop_flag_ || !task_queue_.empty(); 
                });
                
                if (stop_flag_ && task_queue_.empty()) {
                    break;
                }
                
                task = std::move(task_queue_.front());
                task_queue_.pop();
            }
            
            task();
        }
    }
    
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_flag_;
};

/**
 * @brief Event types for integration callbacks
 */
enum class IntegrationEvent {
    StepCompleted,
    StateChanged,
    ParameterUpdated,
    EmergencyStop
};

/**
 * @brief Simple event data structure
 */
template<typename T>
struct Event {
    IntegrationEvent type;
    T data;
    std::chrono::steady_clock::time_point timestamp;
    
    Event(IntegrationEvent t, T d) 
        : type(t), data(std::move(d)), timestamp(std::chrono::steady_clock::now()) {}
};

/**
 * @brief Lightweight async integrator wrapper
 * 
 * This provides async capabilities without heavy communication dependencies.
 * It uses only standard C++ facilities and focuses on the core integration
 * functionality with minimal external dependencies.
 */
template<system_state S, can_be_time T = double>
class AsyncIntegrator {
public:
    using base_integrator_type = AbstractIntegrator<S, T>;
    using state_type = typename base_integrator_type::state_type;
    using time_type = typename base_integrator_type::time_type;
    using value_type = typename base_integrator_type::value_type;
    using system_function = typename base_integrator_type::system_function;
    
    // Event callback types
    using step_callback = std::function<void(const state_type&, time_type)>;
    using parameter_callback = std::function<void(const std::string&, double)>;
    using emergency_callback = std::function<void()>;
    
    /**
     * @brief Configuration for async operation
     */
    struct Config {
        bool enable_async_stepping = false;
        bool enable_state_monitoring = false;
        std::chrono::microseconds monitoring_interval{1000};
        size_t max_concurrent_operations = 4;
    };
    
    explicit AsyncIntegrator(
        std::unique_ptr<base_integrator_type> integrator,
        Config config = {}
    ) : base_integrator_(std::move(integrator))
      , config_(config)
      , executor_(config.max_concurrent_operations)
      , running_(false)
      , emergency_stop_(false) {}
    
    ~AsyncIntegrator() {
        stop();
    }
    
    /**
     * @brief Start async operation
     */
    void start() {
        if (running_.exchange(true)) {
            return;
        }
        
        if (config_.enable_state_monitoring) {
            monitoring_thread_ = std::thread([this] { monitoring_loop(); });
        }
    }
    
    /**
     * @brief Stop async operation
     */
    void stop() {
        if (!running_.exchange(false)) {
            return;
        }
        
        monitoring_condition_.notify_all();
        
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
        
        executor_.shutdown();
    }
    
    /**
     * @brief Async integration step
     */
    std::future<void> step_async(state_type& state, time_type dt) {
        if (emergency_stop_.load()) {
            auto promise = std::promise<void>();
            promise.set_exception(std::make_exception_ptr(
                std::runtime_error("Emergency stop activated")));
            return promise.get_future();
        }
        
        return executor_.submit([this, &state, dt]() {
            std::lock_guard<std::mutex> lock(integration_mutex_);
            base_integrator_->step(state, dt);
            
            // Notify step completion
            if (step_callback_) {
                step_callback_(state, base_integrator_->current_time());
            }
        });
    }
    
    /**
     * @brief Async integration over time interval
     */
    std::future<void> integrate_async(state_type& state, time_type dt, time_type end_time) {
        return executor_.submit([this, &state, dt, end_time]() {
            while (base_integrator_->current_time() < end_time && !emergency_stop_.load()) {
                time_type step_size = std::min(dt, end_time - base_integrator_->current_time());
                
                {
                    std::lock_guard<std::mutex> lock(integration_mutex_);
                    base_integrator_->step(state, step_size);
                }
                
                // Notify step completion
                if (step_callback_) {
                    step_callback_(state, base_integrator_->current_time());
                }
                
                // Allow other operations
                std::this_thread::yield();
            }
        });
    }
    
    /**
     * @brief Synchronous delegation to base integrator
     */
    void step(state_type& state, time_type dt) {
        if (emergency_stop_.load()) {
            throw std::runtime_error("Emergency stop activated");
        }
        
        std::lock_guard<std::mutex> lock(integration_mutex_);
        base_integrator_->step(state, dt);
        
        if (step_callback_) {
            step_callback_(state, base_integrator_->current_time());
        }
    }
    
    void integrate(state_type& state, time_type dt, time_type end_time) {
        if (!running_.load() && config_.enable_async_stepping) {
            start();
        }
        
        base_integrator_->integrate(state, dt, end_time);
    }
    
    // Getters/Setters
    time_type current_time() const { return base_integrator_->current_time(); }
    void set_time(time_type t) { base_integrator_->set_time(t); }
    void set_system(system_function sys) { base_integrator_->set_system(std::move(sys)); }
    
    /**
     * @brief Register callbacks for different events
     */
    void set_step_callback(step_callback callback) {
        step_callback_ = std::move(callback);
    }
    
    void set_parameter_callback(parameter_callback callback) {
        parameter_callback_ = std::move(callback);
    }
    
    void set_emergency_callback(emergency_callback callback) {
        emergency_callback_ = std::move(callback);
    }
    
    /**
     * @brief Update parameter asynchronously
     */
    std::future<void> update_parameter_async(const std::string& name, double value) {
        return executor_.submit([this, name, value]() {
            if (parameter_callback_) {
                parameter_callback_(name, value);
            }
        });
    }
    
    /**
     * @brief Emergency stop
     */
    void emergency_stop() {
        emergency_stop_.store(true);
        if (emergency_callback_) {
            emergency_callback_();
        }
    }
    
    /**
     * @brief Reset emergency stop
     */
    void reset_emergency_stop() {
        emergency_stop_.store(false);
    }
    
    /**
     * @brief Get current state (thread-safe)
     */
    state_type get_current_state() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return current_state_;
    }

private:
    std::unique_ptr<base_integrator_type> base_integrator_;
    Config config_;
    AsyncExecutor executor_;
    
    std::atomic<bool> running_;
    std::atomic<bool> emergency_stop_;
    
    mutable std::mutex integration_mutex_;
    mutable std::mutex state_mutex_;
    state_type current_state_;
    
    std::thread monitoring_thread_;
    std::condition_variable monitoring_condition_;
    
    // Callbacks
    step_callback step_callback_;
    parameter_callback parameter_callback_;
    emergency_callback emergency_callback_;
    
    void monitoring_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            monitoring_condition_.wait_for(
                lock,
                config_.monitoring_interval,
                [this] { return !running_.load(); }
            );
            
            if (!running_.load()) break;
            
            // Update monitored state
            current_state_ = get_integration_state();
        }
    }
    
    state_type get_integration_state() const {
        // This would need to be implemented based on the specific integrator
        // For now, return a default-constructed state
        if constexpr (std::is_default_constructible_v<state_type>) {
            return state_type{};
        } else {
            throw std::runtime_error("Cannot get integration state - state type not default constructible");
        }
    }
};

/**
 * @brief Factory functions for creating async integrators
 */
namespace factory {

template<system_state S, can_be_time T = double>
auto make_async_rk45(
    typename AbstractIntegrator<S, T>::system_function sys,
    typename AsyncIntegrator<S, T>::Config config = {},
    T rtol = static_cast<T>(1e-6),
    T atol = static_cast<T>(1e-9)
) {
    auto base = std::make_unique<diffeq::integrators::ode::RK45Integrator<S, T>>(std::move(sys), rtol, atol);
    return std::make_unique<AsyncIntegrator<S, T>>(std::move(base), config);
}

template<system_state S, can_be_time T = double>
auto make_async_dop853(
    typename AbstractIntegrator<S, T>::system_function sys,
    typename AsyncIntegrator<S, T>::Config config = {},
    T rtol = static_cast<T>(1e-10),
    T atol = static_cast<T>(1e-15)
) {
    auto base = std::make_unique<diffeq::integrators::ode::DOP853Integrator<S, T>>(std::move(sys), rtol, atol);
    return std::make_unique<AsyncIntegrator<S, T>>(std::move(base), config);
}

template<system_state S, can_be_time T = double>
auto make_async_bdf(
    typename AbstractIntegrator<S, T>::system_function sys,
    typename AsyncIntegrator<S, T>::Config config = {},
    T rtol = static_cast<T>(1e-6),
    T atol = static_cast<T>(1e-9)
) {
    auto base = std::make_unique<diffeq::integrators::ode::BDFIntegrator<S, T>>(std::move(sys), rtol, atol);
    return std::make_unique<AsyncIntegrator<S, T>>(std::move(base), config);
}

} // namespace factory

/**
 * @brief Convenience functions for common async patterns
 */
template<typename F, typename... Args>
auto async_execute(F&& func, Args&&... args) {
    static AsyncExecutor global_executor;
    return global_executor.submit([f = std::forward<F>(func), 
                                  args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        return std::apply(std::move(f), std::move(args_tuple));
    });
}

#if DIFFEQ_HAS_STD_EXECUTION
/**
 * @brief Use std::execution when available
 */
template<typename ExecutionPolicy, typename F, typename... Args>
auto execute_std(ExecutionPolicy&& policy, F&& func, Args&&... args) {
    // Note: std::execution::execute is not yet standardized
    // This is a placeholder for future C++ standard versions
    // For now, we use our own async executor
    static AsyncExecutor global_executor;
    return global_executor.submit([f = std::forward<F>(func), 
                                  args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        return std::apply(std::move(f), std::move(args_tuple));
    });
}
#endif

} // namespace diffeq::async
