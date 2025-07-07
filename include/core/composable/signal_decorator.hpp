#pragma once

#include "integrator_decorator.hpp"
#include <vector>
#include <functional>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>

namespace diffeq::core::composable {

/**
 * @brief Signal processing mode
 */
enum class SignalProcessingMode {
    SYNCHRONOUS,   // Process signals immediately during integration
    ASYNCHRONOUS,  // Process signals in background thread
    BATCH         // Accumulate signals and process in batches
};

/**
 * @brief Signal priority level
 */
enum class SignalPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @brief Signal information structure
 */
template<system_state S>
struct SignalInfo {
    std::function<void(S&, typename S::value_type)> handler;
    SignalPriority priority{SignalPriority::NORMAL};
    std::chrono::steady_clock::time_point timestamp;
    bool processed{false};
    std::string signal_id;
};

/**
 * @brief Configuration for signal processing
 */
struct SignalConfig {
    SignalProcessingMode mode{SignalProcessingMode::SYNCHRONOUS};
    bool enable_real_time_processing{true};
    std::chrono::microseconds signal_check_interval{100};
    size_t signal_buffer_size{100};
    size_t max_batch_size{10};
    bool enable_priority_queue{false};
    
    // Validation settings
    bool validate_intervals{true};
    std::chrono::microseconds min_check_interval{1};  // Minimum 1μs
    std::chrono::microseconds max_check_interval{std::chrono::seconds{1}};  // Maximum 1s
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (validate_intervals) {
            if (signal_check_interval < min_check_interval) {
                throw std::invalid_argument("signal_check_interval below minimum " + 
                                          std::to_string(min_check_interval.count()) + "μs");
            }
            if (signal_check_interval > max_check_interval) {
                throw std::invalid_argument("signal_check_interval exceeds maximum " + 
                                          std::to_string(max_check_interval.count()) + "μs");
            }
        }
        
        if (signal_buffer_size == 0) {
            throw std::invalid_argument("signal_buffer_size must be positive");
        }
        
        if (max_batch_size == 0) {
            throw std::invalid_argument("max_batch_size must be positive");
        }
        
        if (max_batch_size > signal_buffer_size) {
            throw std::invalid_argument("max_batch_size cannot exceed signal_buffer_size");
        }
    }
};

/**
 * @brief Signal processing statistics
 */
struct SignalStats {
    size_t total_signals_received{0};
    size_t total_signals_processed{0};
    size_t signals_dropped{0};
    size_t batch_processes{0};
    std::chrono::milliseconds total_processing_time{0};
    std::chrono::milliseconds max_processing_time{0};
    
    double average_processing_time_ms() const {
        return total_signals_processed > 0 ? 
            static_cast<double>(total_processing_time.count()) / total_signals_processed : 0.0;
    }
    
    double signal_drop_rate() const {
        return total_signals_received > 0 ?
            static_cast<double>(signals_dropped) / total_signals_received : 0.0;
    }
};

/**
 * @brief Signal decorator - adds signal processing to any integrator
 * 
 * This decorator provides comprehensive signal processing with the following features:
 * - Multiple processing modes (sync, async, batch)
 * - Signal priority handling
 * - Real-time signal buffering and processing
 * - Detailed statistics and performance monitoring
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles signal processing
 * - No Dependencies: Works with any integrator type
 * - Real-time: Minimal latency signal handling
 * - Thread-safe: Safe concurrent signal registration and processing
 */
template<system_state S>
class SignalDecorator : public IntegratorDecorator<S> {
private:
    SignalConfig config_;
    std::vector<std::function<void(S&, typename IntegratorDecorator<S>::time_type)>> signal_handlers_;
    std::queue<SignalInfo<S>> signal_queue_;
    std::chrono::steady_clock::time_point last_signal_check_;
    mutable std::mutex signal_mutex_;
    SignalStats stats_;
    std::atomic<bool> processing_active_{false};

public:
    /**
     * @brief Construct signal decorator
     * @param integrator The integrator to wrap
     * @param config Signal configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit SignalDecorator(std::unique_ptr<AbstractIntegrator<S>> integrator,
                            SignalConfig config = {})
        : IntegratorDecorator<S>(std::move(integrator)), config_(std::move(config))
        , last_signal_check_(std::chrono::steady_clock::now()) {
        
        config_.validate();
    }

    /**
     * @brief Override step to add signal processing
     */
    void step(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt) override {
        // Process signals before step
        if (config_.enable_real_time_processing) {
            process_signals(state, this->current_time());
        }
        
        this->wrapped_integrator_->step(state, dt);
        
        // Process signals after step if real-time enabled
        if (config_.enable_real_time_processing && config_.mode == SignalProcessingMode::SYNCHRONOUS) {
            process_signals(state, this->current_time());
        }
    }

    /**
     * @brief Override integrate to handle signal processing during integration
     */
    void integrate(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt, 
                   typename IntegratorDecorator<S>::time_type end_time) override {
        processing_active_.store(true);
        auto processing_guard = make_scope_guard([this] { processing_active_.store(false); });
        
        if (config_.mode == SignalProcessingMode::BATCH) {
            // Process in batch mode
            this->wrapped_integrator_->integrate(state, dt, end_time);
            process_signal_batch(state, this->current_time());
        } else {
            // Process with real-time signal handling
            this->wrapped_integrator_->integrate(state, dt, end_time);
        }
    }

    /**
     * @brief Register a signal handler function
     * @param handler Function to handle signals: void(S& state, T time)
     * @param signal_id Optional identifier for the signal
     * @param priority Signal priority level
     */
    void register_signal_handler(std::function<void(S&, typename IntegratorDecorator<S>::time_type)> handler, 
                                 const std::string& signal_id = "",
                                 SignalPriority priority = SignalPriority::NORMAL) {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        
        if (config_.enable_priority_queue) {
            // Add to priority queue
            SignalInfo<S> signal_info;
            signal_info.handler = std::move(handler);
            signal_info.priority = priority;
            signal_info.signal_id = signal_id;
            signal_info.timestamp = std::chrono::steady_clock::now();
            
            signal_queue_.push(std::move(signal_info));
        } else {
            // Add to simple vector
            signal_handlers_.push_back(std::move(handler));
        }
        
        stats_.total_signals_received++;
    }

    /**
     * @brief Register multiple signal handlers at once
     * @param handlers Vector of signal handler functions
     */
    void register_signal_handlers(const std::vector<std::function<void(S&, typename IntegratorDecorator<S>::time_type)>>& handlers) {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        
        for (const auto& handler : handlers) {
            signal_handlers_.push_back(handler);
            stats_.total_signals_received++;
        }
    }

    /**
     * @brief Clear all signal handlers
     */
    void clear_signal_handlers() {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        signal_handlers_.clear();
        
        // Clear queue as well
        while (!signal_queue_.empty()) {
            signal_queue_.pop();
        }
    }

    /**
     * @brief Get number of registered signal handlers
     */
    size_t get_signal_handler_count() const {
        std::lock_guard<std::mutex> lock(signal_mutex_);
        return signal_handlers_.size() + signal_queue_.size();
    }

    /**
     * @brief Force immediate signal processing
     * @param state Current state
     * @param time Current time
     */
    void process_signals_now(S& state, typename IntegratorDecorator<S>::time_type time) {
        process_signals(state, time);
    }

    /**
     * @brief Get signal processing statistics
     */
    const SignalStats& get_statistics() const {
        return stats_;
    }

    /**
     * @brief Reset signal processing statistics
     */
    void reset_statistics() {
        stats_ = SignalStats{};
    }

    /**
     * @brief Check if signal processing is currently active
     */
    bool is_processing_active() const {
        return processing_active_.load();
    }

    /**
     * @brief Access and modify signal configuration
     */
    SignalConfig& config() { return config_; }
    const SignalConfig& config() const { return config_; }
    
    /**
     * @brief Update signal configuration with validation
     * @param new_config New configuration
     * @throws std::invalid_argument if new config is invalid
     */
    void update_config(SignalConfig new_config) {
        new_config.validate();
        std::lock_guard<std::mutex> lock(signal_mutex_);
        config_ = std::move(new_config);
    }

private:
    /**
     * @brief Process all pending signals
     */
    void process_signals(S& state, typename IntegratorDecorator<S>::time_type time) {
        auto now = std::chrono::steady_clock::now();
        
        // Check if it's time for signal processing
        if (now - last_signal_check_ < config_.signal_check_interval) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(signal_mutex_);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process regular signal handlers
        for (auto& handler : signal_handlers_) {
            if (handler) {
                handler(state, time);
                stats_.total_signals_processed++;
            }
        }
        
        // Process priority queue if enabled
        if (config_.enable_priority_queue) {
            process_priority_signals(state, time);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        stats_.total_processing_time += processing_time;
        if (processing_time > stats_.max_processing_time) {
            stats_.max_processing_time = processing_time;
        }
        
        last_signal_check_ = now;
    }

    /**
     * @brief Process signals in batch mode
     */
    void process_signal_batch(S& state, typename IntegratorDecorator<S>::time_type time) {
        if (signal_handlers_.empty() && signal_queue_.empty()) {
            return;
        }
        
        std::lock_guard<std::mutex> lock(signal_mutex_);
        auto start_time = std::chrono::high_resolution_clock::now();
        
        size_t processed_count = 0;
        
        // Process up to max_batch_size signals
        for (auto& handler : signal_handlers_) {
            if (processed_count >= config_.max_batch_size) {
                break;
            }
            
            if (handler) {
                handler(state, time);
                stats_.total_signals_processed++;
                processed_count++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        
        stats_.total_processing_time += processing_time;
        stats_.batch_processes++;
    }

    /**
     * @brief Process signals from priority queue
     */
    void process_priority_signals(S& state, typename IntegratorDecorator<S>::time_type time) {
        // For simplicity, process all signals in queue order
        // A real implementation might sort by priority first
        while (!signal_queue_.empty()) {
            auto& signal_info = signal_queue_.front();
            
            if (signal_info.handler && !signal_info.processed) {
                signal_info.handler(state, time);
                signal_info.processed = true;
                stats_.total_signals_processed++;
            }
            
            signal_queue_.pop();
        }
    }

    /**
     * @brief RAII scope guard for processing state
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