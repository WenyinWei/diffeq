#pragma once

#include "concepts.hpp"
#include "abstract_integrator.hpp"
#include <memory>
#include <functional>
#include <chrono>
#include <future>
#include <vector>
#include <execution>
#include <thread>

namespace diffeq::core::composable {

/**
 * @brief Base decorator interface for integrator enhancements
 * 
 * This provides the foundation for the decorator pattern, allowing
 * facilities to be stacked independently without tight coupling.
 */
template<system_state S, can_be_time T = double>
class IntegratorDecorator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using system_function = typename base_type::system_function;

protected:
    std::unique_ptr<base_type> wrapped_integrator_;

public:
    explicit IntegratorDecorator(std::unique_ptr<base_type> integrator)
        : base_type(integrator->sys_), wrapped_integrator_(std::move(integrator)) {}

    // Delegate core functionality by default
    void step(state_type& state, time_type dt) override {
        wrapped_integrator_->step(state, dt);
    }

    void integrate(state_type& state, time_type dt, time_type end_time) override {
        wrapped_integrator_->integrate(state, dt, end_time);
    }

    time_type current_time() const override {
        return wrapped_integrator_->current_time();
    }

    void set_time(time_type t) override {
        wrapped_integrator_->set_time(t);
        this->current_time_ = t;
    }

    void set_system(system_function sys) override {
        wrapped_integrator_->set_system(std::move(sys));
        this->sys_ = wrapped_integrator_->sys_;
    }

    // Access to wrapped integrator for advanced use
    base_type& wrapped() { return *wrapped_integrator_; }
    const base_type& wrapped() const { return *wrapped_integrator_; }
};

// ============================================================================
// TIMEOUT FACILITY (Independent, reusable)
// ============================================================================

/**
 * @brief Timeout configuration
 */
struct TimeoutConfig {
    std::chrono::milliseconds timeout_duration{5000};
    bool throw_on_timeout{true};
    bool enable_progress_callback{false};
    std::chrono::milliseconds progress_interval{100};
    std::function<bool(double, double, std::chrono::milliseconds)> progress_callback;
};

/**
 * @brief Timeout result information
 */
struct TimeoutResult {
    bool completed{false};
    std::chrono::milliseconds elapsed_time{0};
    double final_time{0.0};
    std::string error_message;
    
    bool is_success() const { return completed && error_message.empty(); }
    bool is_timeout() const { return !completed && error_message.find("timeout") != std::string::npos; }
};

/**
 * @brief Timeout decorator - adds timeout protection to any integrator
 */
template<system_state S, can_be_time T = double>
class TimeoutDecorator : public IntegratorDecorator<S, T> {
private:
    TimeoutConfig config_;

public:
    explicit TimeoutDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator, 
                             TimeoutConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {}

    TimeoutResult integrate_with_timeout(typename IntegratorDecorator<S, T>::state_type& state, 
                                        T dt, T end_time) {
        const auto start_time = std::chrono::high_resolution_clock::now();
        TimeoutResult result;
        result.final_time = this->current_time();

        try {
            auto future = std::async(std::launch::async, [this, &state, dt, end_time]() {
                this->wrapped_integrator_->integrate(state, dt, end_time);
            });

            if (future.wait_for(config_.timeout_duration) == std::future_status::timeout) {
                result.completed = false;
                result.error_message = "Integration timed out after " + 
                                     std::to_string(config_.timeout_duration.count()) + "ms";
            } else {
                future.get();
                result.completed = true;
            }
            
            result.final_time = this->current_time();
        } catch (const std::exception& e) {
            result.completed = false;
            result.error_message = "Integration failed: " + std::string(e.what());
        }

        const auto end_time_clock = std::chrono::high_resolution_clock::now();
        result.elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time_clock - start_time);

        if (!result.completed && config_.throw_on_timeout && result.is_timeout()) {
            throw std::runtime_error(result.error_message);
        }

        return result;
    }

    TimeoutConfig& config() { return config_; }
    const TimeoutConfig& config() const { return config_; }
};

// ============================================================================
// PARALLEL EXECUTION FACILITY (Independent, reusable)
// ============================================================================

/**
 * @brief Parallel execution configuration
 */
struct ParallelConfig {
    size_t max_threads{0};  // 0 = auto-detect
    size_t chunk_size{1};
    bool enable_auto_chunking{true};
};

/**
 * @brief Parallel execution decorator - adds batch processing to any integrator
 */
template<system_state S, can_be_time T = double>
class ParallelDecorator : public IntegratorDecorator<S, T> {
private:
    ParallelConfig config_;

public:
    explicit ParallelDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                              ParallelConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {
        
        if (config_.max_threads == 0) {
            config_.max_threads = std::thread::hardware_concurrency();
        }
    }

    template<typename StateRange>
    void integrate_batch(StateRange&& states, T dt, T end_time) {
        std::for_each(std::execution::par_unseq, states.begin(), states.end(),
            [this, dt, end_time](auto& state) {
                // Create thread-local copy of integrator
                auto local_integrator = this->create_copy();
                local_integrator->integrate(state, dt, end_time);
            });
    }

    template<typename Generator, typename Processor>
    auto integrate_monte_carlo(size_t num_simulations, Generator&& generator, 
                              Processor&& processor, T dt, T end_time) {
        using result_type = std::invoke_result_t<Processor, S>;
        std::vector<result_type> results(num_simulations);

        std::for_each(std::execution::par_unseq,
            std::views::iota(0UL, num_simulations).begin(),
            std::views::iota(0UL, num_simulations).end(),
            [&](size_t i) {
                auto local_integrator = this->create_copy();
                auto state = generator(i);
                local_integrator->integrate(state, dt, end_time);
                results[i] = processor(state);
            });

        return results;
    }

    ParallelConfig& config() { return config_; }
    const ParallelConfig& config() const { return config_; }

private:
    std::unique_ptr<AbstractIntegrator<S, T>> create_copy() {
        // This would need to be implemented based on integrator factory pattern
        // For now, throw to indicate need for implementation
        throw std::runtime_error("Integrator copying not implemented - need factory pattern");
    }
};

// ============================================================================
// ASYNC EXECUTION FACILITY (Independent, reusable) 
// ============================================================================

/**
 * @brief Async execution configuration
 */
struct AsyncConfig {
    size_t thread_pool_size{0};  // 0 = auto-detect
    bool enable_progress_monitoring{false};
    std::chrono::microseconds monitoring_interval{1000};
};

/**
 * @brief Async execution decorator - adds async capabilities to any integrator
 */
template<system_state S, can_be_time T = double>
class AsyncDecorator : public IntegratorDecorator<S, T> {
private:
    AsyncConfig config_;

public:
    explicit AsyncDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                           AsyncConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {}

    std::future<void> integrate_async(typename IntegratorDecorator<S, T>::state_type& state, 
                                     T dt, T end_time) {
        return std::async(std::launch::async, [this, &state, dt, end_time]() {
            this->wrapped_integrator_->integrate(state, dt, end_time);
        });
    }

    std::future<void> step_async(typename IntegratorDecorator<S, T>::state_type& state, T dt) {
        return std::async(std::launch::async, [this, &state, dt]() {
            this->wrapped_integrator_->step(state, dt);
        });
    }

    AsyncConfig& config() { return config_; }
    const AsyncConfig& config() const { return config_; }
};

// ============================================================================
// OUTPUT HANDLING FACILITY (Independent, reusable)
// ============================================================================

/**
 * @brief Output mode enumeration
 */
enum class OutputMode {
    ONLINE,     // Real-time output during integration
    OFFLINE,    // Buffered output after integration
    HYBRID      // Combination of online and offline
};

/**
 * @brief Output configuration
 */
struct OutputConfig {
    OutputMode mode{OutputMode::ONLINE};
    std::chrono::microseconds output_interval{1000};
    size_t buffer_size{1000};
    bool enable_compression{false};
};

/**
 * @brief Output decorator - adds configurable output to any integrator
 */
template<system_state S, can_be_time T = double>
class OutputDecorator : public IntegratorDecorator<S, T> {
private:
    OutputConfig config_;
    std::function<void(const S&, T, size_t)> output_handler_;
    std::vector<std::pair<S, T>> output_buffer_;
    std::chrono::steady_clock::time_point last_output_;
    size_t step_count_{0};

public:
    explicit OutputDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                            OutputConfig config = {},
                            std::function<void(const S&, T, size_t)> handler = nullptr)
        : IntegratorDecorator<S, T>(std::move(integrator))
        , config_(std::move(config))
        , output_handler_(std::move(handler))
        , last_output_(std::chrono::steady_clock::now()) {}

    void step(typename IntegratorDecorator<S, T>::state_type& state, T dt) override {
        this->wrapped_integrator_->step(state, dt);
        ++step_count_;
        
        handle_output(state, this->current_time());
    }

    void integrate(typename IntegratorDecorator<S, T>::state_type& state, T dt, T end_time) override {
        if (config_.mode == OutputMode::OFFLINE) {
            // Just integrate and buffer final result
            this->wrapped_integrator_->integrate(state, dt, end_time);
            buffer_output(state, this->current_time());
        } else {
            // Step-by-step with online output
            while (this->current_time() < end_time) {
                T step_size = std::min(dt, end_time - this->current_time());
                this->step(state, step_size);
            }
        }
        
        if (config_.mode == OutputMode::OFFLINE || config_.mode == OutputMode::HYBRID) {
            flush_output();
        }
    }

    void set_output_handler(std::function<void(const S&, T, size_t)> handler) {
        output_handler_ = std::move(handler);
    }

    const std::vector<std::pair<S, T>>& get_buffer() const { return output_buffer_; }
    
    void clear_buffer() { output_buffer_.clear(); }

    OutputConfig& config() { return config_; }
    const OutputConfig& config() const { return config_; }

private:
    void handle_output(const S& state, T time) {
        auto now = std::chrono::steady_clock::now();
        
        if (config_.mode == OutputMode::ONLINE || config_.mode == OutputMode::HYBRID) {
            if (now - last_output_ >= config_.output_interval) {
                if (output_handler_) {
                    output_handler_(state, time, step_count_);
                }
                last_output_ = now;
            }
        }
        
        if (config_.mode == OutputMode::OFFLINE || config_.mode == OutputMode::HYBRID) {
            buffer_output(state, time);
        }
    }

    void buffer_output(const S& state, T time) {
        if (output_buffer_.size() >= config_.buffer_size) {
            output_buffer_.erase(output_buffer_.begin());
        }
        output_buffer_.emplace_back(state, time);
    }

    void flush_output() {
        if (output_handler_) {
            for (const auto& [state, time] : output_buffer_) {
                output_handler_(state, time, step_count_);
            }
        }
    }
};

// ============================================================================
// SIGNAL PROCESSING FACILITY (Independent, reusable)
// ============================================================================

/**
 * @brief Signal processing configuration
 */
struct SignalConfig {
    bool enable_real_time_processing{true};
    std::chrono::microseconds signal_check_interval{100};
    size_t signal_buffer_size{100};
};

/**
 * @brief Signal decorator - adds signal processing to any integrator
 */
template<system_state S, can_be_time T = double>
class SignalDecorator : public IntegratorDecorator<S, T> {
private:
    SignalConfig config_;
    std::vector<std::function<void(S&, T)>> signal_handlers_;

public:
    explicit SignalDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                            SignalConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config)) {}

    void step(typename IntegratorDecorator<S, T>::state_type& state, T dt) override {
        // Process signals before step
        process_signals(state, this->current_time());
        
        this->wrapped_integrator_->step(state, dt);
        
        // Process signals after step if real-time enabled
        if (config_.enable_real_time_processing) {
            process_signals(state, this->current_time());
        }
    }

    void register_signal_handler(std::function<void(S&, T)> handler) {
        signal_handlers_.push_back(std::move(handler));
    }

    void clear_signal_handlers() {
        signal_handlers_.clear();
    }

    SignalConfig& config() { return config_; }
    const SignalConfig& config() const { return config_; }

private:
    void process_signals(S& state, T time) {
        for (auto& handler : signal_handlers_) {
            handler(state, time);
        }
    }
};

// ============================================================================
// COMPOSITION BUILDER (Flexible combination)
// ============================================================================

/**
 * @brief Builder for composing multiple facilities
 * 
 * This allows flexible combination of any facilities without
 * exponential class combinations.
 */
template<system_state S, can_be_time T = double>
class IntegratorBuilder {
private:
    std::unique_ptr<AbstractIntegrator<S, T>> integrator_;

public:
    explicit IntegratorBuilder(std::unique_ptr<AbstractIntegrator<S, T>> integrator)
        : integrator_(std::move(integrator)) {}

    IntegratorBuilder& with_timeout(TimeoutConfig config = {}) {
        integrator_ = std::make_unique<TimeoutDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    IntegratorBuilder& with_parallel(ParallelConfig config = {}) {
        integrator_ = std::make_unique<ParallelDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    IntegratorBuilder& with_async(AsyncConfig config = {}) {
        integrator_ = std::make_unique<AsyncDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    IntegratorBuilder& with_output(OutputConfig config = {}, 
                                  std::function<void(const S&, T, size_t)> handler = nullptr) {
        integrator_ = std::make_unique<OutputDecorator<S, T>>(
            std::move(integrator_), std::move(config), std::move(handler));
        return *this;
    }

    IntegratorBuilder& with_signals(SignalConfig config = {}) {
        integrator_ = std::make_unique<SignalDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    std::unique_ptr<AbstractIntegrator<S, T>> build() {
        return std::move(integrator_);
    }

    // Convenience method to get specific decorator types
    template<typename DecoratorType>
    DecoratorType* get_as() {
        return dynamic_cast<DecoratorType*>(integrator_.get());
    }
};

// ============================================================================
// FACTORY FUNCTIONS (Easy creation)
// ============================================================================

/**
 * @brief Create a builder starting with any integrator
 */
template<system_state S, can_be_time T = double>
auto make_builder(std::unique_ptr<AbstractIntegrator<S, T>> integrator) {
    return IntegratorBuilder<S, T>(std::move(integrator));
}

/**
 * @brief Convenience functions for common combinations
 */
template<system_state S, can_be_time T = double>
auto with_timeout_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator, 
                      TimeoutConfig config = {}) {
    return make_builder(std::move(integrator)).with_timeout(std::move(config)).build();
}

template<system_state S, can_be_time T = double>
auto with_parallel_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                       ParallelConfig config = {}) {
    return make_builder(std::move(integrator)).with_parallel(std::move(config)).build();
}

template<system_state S, can_be_time T = double>
auto with_async_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                     AsyncConfig config = {}) {
    return make_builder(std::move(integrator)).with_async(std::move(config)).build();
}

} // namespace diffeq::core::composable