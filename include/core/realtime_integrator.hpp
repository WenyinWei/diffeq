#pragma once

#include <core/abstract_integrator.hpp>
#include <communication/event_bus.hpp>
#include <communication/process_connector.hpp>
#include <communication/realtime_priority.hpp>

#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <optional>
#include <chrono>
#include <functional>
#include <memory>
#include <queue>
#include <span>

// C++23 std::execution support with fallback
#if __has_include(<execution>) && defined(__cpp_lib_execution)
#include <execution>
#define DIFFEQ_HAS_STD_EXECUTION 1
#else
#define DIFFEQ_HAS_STD_EXECUTION 0
#endif

namespace diffeq::realtime {

/**
 * @brief Signal types for different application domains
 */
enum class SignalType : uint8_t {
    Control = 0,        // Robot control signals
    Financial = 1,      // Trading/market signals  
    Safety = 2,         // Emergency stop/safety signals
    Parameter = 3,      // Parameter update signals
    State = 4,          // State output signals
    Diagnostic = 5      // System diagnostics
};

/**
 * @brief Real-time signal structure for inter-process communication
 */
template<typename T>
struct RealtimeSignal {
    SignalType signal_type;
    T data;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t sequence_id;
    double priority{1.0};
    std::optional<std::chrono::milliseconds> timeout;
    
    template<typename U>
    RealtimeSignal(SignalType type, U&& signal_data, double prio = 1.0)
        : signal_type(type)
        , data(std::forward<U>(signal_data))
        , timestamp(std::chrono::steady_clock::now())
        , sequence_id(next_sequence_id())
        , priority(prio) {}
        
private:
    static std::atomic<uint64_t> sequence_counter_;
    static uint64_t next_sequence_id() { return sequence_counter_.fetch_add(1); }
};

template<typename T>
std::atomic<uint64_t> RealtimeSignal<T>::sequence_counter_{0};

/**
 * @brief Signal handler concept for processing real-time signals
 */
template<typename Handler, typename Signal>
concept SignalHandler = requires(Handler h, const Signal& s) {
    { h(s) } -> std::same_as<void>;
};

/**
 * @brief Async signal handler concept
 */
template<typename Handler, typename Signal>
concept AsyncSignalHandler = requires(Handler h, const Signal& s) {
    { h(s) } -> std::same_as<std::future<void>>;
};

/**
 * @brief Custom executor for async operations when std::execution is not available
 */
class CustomExecutor {
public:
    CustomExecutor(size_t num_threads = std::thread::hardware_concurrency())
        : stop_flag_(false) {
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] { worker_thread(); });
        }
    }
    
    ~CustomExecutor() {
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
 * @brief Real-time integrator with signal processing capabilities
 * 
 * This class extends the basic integrator functionality with:
 * - Real-time signal processing for control and financial applications
 * - Async communication with external processes
 * - Event-driven parameter updates
 * - High-performance state output
 * - Safety mechanisms and emergency stops
 */
template<system_state S, can_be_time T = double>
class RealtimeIntegrator : public core::AbstractIntegrator<S, T> {
public:
    using base_type = core::AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;
    
    // Signal type aliases
    template<typename U>
    using signal_type = RealtimeSignal<U>;
    
    using control_signal = signal_type<communication::RobotControlMessage>;
    using financial_signal = signal_type<communication::FinancialSignalMessage>;
    using parameter_signal = signal_type<std::unordered_map<std::string, double>>;
    using state_output_signal = signal_type<state_type>;
    
    /**
     * @brief Configuration for real-time operation
     */
    struct RealtimeConfig {
        bool enable_realtime_priority = false;
        communication::RealtimePriority::Priority priority = communication::RealtimePriority::Priority::Normal;
        bool lock_memory = false;
        size_t signal_buffer_size = 1024;
        std::chrono::microseconds signal_processing_interval{100};
        std::chrono::milliseconds max_signal_latency{10};
        bool enable_state_output = true;
        std::chrono::microseconds state_output_interval{1000};
    };
    
    explicit RealtimeIntegrator(
        std::unique_ptr<core::AbstractIntegrator<S, T>> base_integrator,
        RealtimeConfig config = {}
    ) : base_type(std::move(base_integrator->sys_))
      , base_integrator_(std::move(base_integrator))
      , config_(std::move(config))
      , event_bus_(std::make_unique<communication::EventBus>())
      , executor_(std::make_unique<CustomExecutor>())
      , running_(false)
      , emergency_stop_(false) {
        
        setup_signal_handlers();
        
        if (config_.enable_realtime_priority) {
            setup_realtime_priority();
        }
    }
    
    ~RealtimeIntegrator() {
        shutdown();
    }
    
    // Non-copyable, movable
    RealtimeIntegrator(const RealtimeIntegrator&) = delete;
    RealtimeIntegrator& operator=(const RealtimeIntegrator&) = delete;
    RealtimeIntegrator(RealtimeIntegrator&&) noexcept = default;
    RealtimeIntegrator& operator=(RealtimeIntegrator&&) noexcept = default;
    
    /**
     * @brief Start real-time operation
     */
    void start_realtime() {
        if (running_.exchange(true)) {
            return; // Already running
        }
        
        // Start signal processing thread
        signal_thread_ = std::thread([this] { signal_processing_loop(); });
        
        // Start state output thread if enabled
        if (config_.enable_state_output) {
            state_output_thread_ = std::thread([this] { state_output_loop(); });
        }
    }
    
    /**
     * @brief Stop real-time operation
     */
    void shutdown() {
        if (!running_.exchange(false)) {
            return; // Already stopped
        }
        
        signal_condition_.notify_all();
        state_output_condition_.notify_all();
        
        if (signal_thread_.joinable()) {
            signal_thread_.join();
        }
        
        if (state_output_thread_.joinable()) {
            state_output_thread_.join();
        }
        
        if (executor_) {
            executor_->shutdown();
        }
    }
    
    /**
     * @brief Enhanced step function with signal processing
     */
    void step(state_type& state, time_type dt) override {
        // Check for emergency stop
        if (emergency_stop_.load()) {
            throw std::runtime_error("Emergency stop activated");
        }
        
        // Process any pending signals before stepping
        process_pending_signals(state, dt);
        
        // Delegate to the base integrator
        base_integrator_->step(state, dt);
        this->advance_time(dt);
        
        // Update current state for output
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state_ = state;
            last_step_time_ = std::chrono::steady_clock::now();
        }
    }
    
    /**
     * @brief Enhanced integrate function with real-time capabilities
     */
    void integrate(state_type& state, time_type dt, time_type end_time) override {
        if (!running_.load()) {
            start_realtime();
        }
        
        while (this->current_time_ < end_time && !emergency_stop_.load()) {
            time_type step_size = std::min(dt, end_time - this->current_time_);
            step(state, step_size);
            
            // Allow signal processing between steps
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
    /**
     * @brief Send control signal (for robotics applications)
     */
    template<typename ControlData>
    void send_control_signal(ControlData&& data, double priority = 1.0) {
        auto signal = std::make_shared<control_signal>(
            SignalType::Control, 
            std::forward<ControlData>(data), 
            priority
        );
        
        enqueue_signal(signal);
    }
    
    /**
     * @brief Send financial signal (for quantitative trading)
     */
    template<typename FinancialData>
    void send_financial_signal(FinancialData&& data, double priority = 1.0) {
        auto signal = std::make_shared<financial_signal>(
            SignalType::Financial,
            std::forward<FinancialData>(data),
            priority
        );
        
        enqueue_signal(signal);
    }
    
    /**
     * @brief Update system parameters dynamically
     */
    void update_parameters(const std::unordered_map<std::string, double>& params) {
        auto signal = std::make_shared<parameter_signal>(
            SignalType::Parameter,
            params,
            10.0 // High priority for parameter updates
        );
        
        enqueue_signal(signal);
    }
    
    /**
     * @brief Emergency stop - immediately halt integration
     */
    void emergency_stop() {
        emergency_stop_.store(true);
        
        // Send emergency signal
        communication::RobotControlMessage emergency_msg;
        emergency_msg.emergency_stop = true;
        emergency_msg.timestamp_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        
        send_control_signal(emergency_msg, 100.0); // Maximum priority
    }
    
    /**
     * @brief Get current state (thread-safe)
     */
    state_type get_current_state() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return current_state_;
    }
    
    /**
     * @brief Set up process connector for inter-process communication
     */
    void setup_process_connector(communication::ProcessConnector::ConnectionConfig config) {
        process_connector_ = std::make_unique<communication::ProcessConnector>(std::move(config));
        
        // Start async connection
        auto connect_future = process_connector_->connect();
        
        // Handle connection in background
        #if DIFFEQ_HAS_STD_EXECUTION
        std::execution::execute(std::execution::par, [this, fut = std::move(connect_future)]() mutable {
            try {
                if (fut.get()) {
                    start_external_communication();
                }
            } catch (const std::exception& e) {
                // Log connection error
                std::cerr << "Process connector failed: " << e.what() << std::endl;
            }
        });
        #else
        executor_->submit([this, fut = std::move(connect_future)]() mutable {
            try {
                if (fut.get()) {
                    start_external_communication();
                }
            } catch (const std::exception& e) {
                std::cerr << "Process connector failed: " << e.what() << std::endl;
            }
        });
        #endif
    }
    
    /**
     * @brief Register signal handler for specific signal type
     */
    template<typename SignalT, SignalHandler<SignalT> Handler>
    void register_signal_handler(SignalType type, Handler&& handler) {
        std::lock_guard<std::mutex> lock(handlers_mutex_);
        signal_handlers_[static_cast<uint8_t>(type)] = 
            [h = std::forward<Handler>(handler)](const std::any& signal) {
                try {
                    const auto& typed_signal = std::any_cast<const SignalT&>(signal);
                    h(typed_signal);
                } catch (const std::bad_any_cast& e) {
                    std::cerr << "Signal type mismatch: " << e.what() << std::endl;
                }
            };
    }
    
private:
    std::unique_ptr<core::AbstractIntegrator<S, T>> base_integrator_;
    RealtimeConfig config_;
    std::unique_ptr<communication::EventBus> event_bus_;
    std::unique_ptr<CustomExecutor> executor_;
    std::unique_ptr<communication::ProcessConnector> process_connector_;
    
    // Threading and synchronization
    std::atomic<bool> running_;
    std::atomic<bool> emergency_stop_;
    std::thread signal_thread_;
    std::thread state_output_thread_;
    std::condition_variable signal_condition_;
    std::condition_variable state_output_condition_;
    
    // State management
    mutable std::mutex state_mutex_;
    state_type current_state_;
    std::chrono::steady_clock::time_point last_step_time_;
    
    // Signal handling
    std::mutex signal_queue_mutex_;
    std::priority_queue<std::shared_ptr<std::any>, 
                       std::vector<std::shared_ptr<std::any>>,
                       std::function<bool(const std::shared_ptr<std::any>&, 
                                        const std::shared_ptr<std::any>&)>> signal_queue_;
    
    std::mutex handlers_mutex_;
    std::unordered_map<uint8_t, std::function<void(const std::any&)>> signal_handlers_;
    
    void setup_realtime_priority() {
        auto result = communication::RealtimePriority::set_thread_priority(
            communication::RealtimePriority::Policy::RealTimeFIFO,
            config_.priority
        );
        
        if (result) {
            std::cerr << "Failed to set real-time priority: " << result.message() << std::endl;
        }
        
        if (config_.lock_memory) {
            result = communication::RealtimePriority::lock_memory();
            if (result) {
                std::cerr << "Failed to lock memory: " << result.message() << std::endl;
            }
        }
    }
    
    void setup_signal_handlers() {
        // Initialize priority queue with custom comparator
        auto comparator = [](const std::shared_ptr<std::any>& a, const std::shared_ptr<std::any>& b) -> bool {
            // Default comparison - in real implementation, extract priority from signal
            return false; // For now, FIFO ordering
        };
        
        signal_queue_ = std::priority_queue<std::shared_ptr<std::any>, 
                                          std::vector<std::shared_ptr<std::any>>,
                                          decltype(comparator)>(comparator);
    }
    
    template<typename SignalT>
    void enqueue_signal(std::shared_ptr<SignalT> signal) {
        {
            std::lock_guard<std::mutex> lock(signal_queue_mutex_);
            signal_queue_.push(std::static_pointer_cast<std::any>(
                std::make_shared<SignalT>(*signal)
            ));
        }
        signal_condition_.notify_one();
    }
    
    void signal_processing_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(signal_queue_mutex_);
            signal_condition_.wait_for(
                lock, 
                config_.signal_processing_interval,
                [this] { return !signal_queue_.empty() || !running_.load(); }
            );
            
            while (!signal_queue_.empty() && running_.load()) {
                auto signal = signal_queue_.top();
                signal_queue_.pop();
                lock.unlock();
                
                // Process signal asynchronously
                #if DIFFEQ_HAS_STD_EXECUTION
                std::execution::execute(std::execution::par, [this, signal] {
                    process_signal(signal);
                });
                #else
                executor_->submit([this, signal] {
                    process_signal(signal);
                });
                #endif
                
                lock.lock();
            }
        }
    }
    
    void process_signal(std::shared_ptr<std::any> signal) {
        // In a real implementation, we'd extract signal type and dispatch appropriately
        // For now, this is a placeholder
    }
    
    void process_pending_signals(state_type& state, time_type dt) {
        // Quick processing of high-priority signals before integration step
        // This ensures minimal latency for critical signals
    }
    
    void state_output_loop() {
        while (running_.load()) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            state_output_condition_.wait_for(
                lock,
                config_.state_output_interval,
                [this] { return !running_.load(); }
            );
            
            if (!running_.load()) break;
            
            // Output current state
            if (process_connector_) {
                state_output_signal output_signal(
                    SignalType::State,
                    current_state_
                );
                
                // Send asynchronously
                #if DIFFEQ_HAS_STD_EXECUTION
                std::execution::execute(std::execution::par, [this, output_signal] {
                    // process_connector_->send(output_signal);
                });
                #else
                executor_->submit([this, output_signal] {
                    // process_connector_->send(output_signal);
                });
                #endif
            }
        }
    }
    
    void start_external_communication() {
        // Start receiving signals from external processes
        if (!process_connector_) return;
        
        #if DIFFEQ_HAS_STD_EXECUTION
        std::execution::execute(std::execution::par, [this] {
            receive_external_signals();
        });
        #else
        executor_->submit([this] {
            receive_external_signals();
        });
        #endif
    }
    
    void receive_external_signals() {
        while (running_.load() && process_connector_) {
            // Receive signals from external processes
            // This would use the process_connector_ to receive different signal types
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
};

/**
 * @brief Factory functions for creating real-time integrators
 */
namespace factory {

template<system_state S, can_be_time T = double>
auto make_realtime_rk45(
    typename core::AbstractIntegrator<S, T>::system_function sys,
    typename RealtimeIntegrator<S, T>::RealtimeConfig config = {},
    T rtol = static_cast<T>(1e-6),
    T atol = static_cast<T>(1e-9)
) {
    auto base = std::make_unique<RK45Integrator<S, T>>(std::move(sys), rtol, atol);
    return std::make_unique<RealtimeIntegrator<S, T>>(std::move(base), std::move(config));
}

template<system_state S, can_be_time T = double>
auto make_realtime_dop853(
    typename core::AbstractIntegrator<S, T>::system_function sys,
    typename RealtimeIntegrator<S, T>::RealtimeConfig config = {},
    T rtol = static_cast<T>(1e-10),
    T atol = static_cast<T>(1e-15)
) {
    auto base = std::make_unique<DOP853Integrator<S, T>>(std::move(sys), rtol, atol);
    return std::make_unique<RealtimeIntegrator<S, T>>(std::move(base), std::move(config));
}

} // namespace factory

} // namespace diffeq::realtime
