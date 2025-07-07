#pragma once

#include "integrator_decorator.hpp"
#include <functional>
#include <vector>
#include <map>
#include <queue>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <algorithm>
#include <utility>

namespace diffeq::core::composable {

/**
 * @brief Event trigger type enumeration
 */
enum class EventTrigger {
    TIME_BASED,         // Trigger at specific time intervals
    STATE_BASED,        // Trigger based on state conditions
    EXTERNAL_SIGNAL,    // Trigger from external source
    SENSOR_DATA,        // Trigger when sensor data arrives
    CONTROL_FEEDBACK,   // Trigger for control loop feedback
    THRESHOLD_CROSSING, // Trigger when value crosses threshold
    DERIVATIVE_CHANGE,  // Trigger based on derivative changes
    CUSTOM              // Custom trigger condition
};

/**
 * @brief Event priority levels
 */
enum class EventPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3,
    EMERGENCY = 4
};

/**
 * @brief Event processing mode
 */
enum class EventProcessingMode {
    IMMEDIATE,          // Process immediately when triggered
    DEFERRED,           // Process at next integration step
    BATCHED,            // Process in batches
    ASYNC               // Process asynchronously
};

/**
 * @brief Event configuration
 */
struct EventConfig {
    EventProcessingMode processing_mode{EventProcessingMode::IMMEDIATE};
    bool enable_priority_queue{true};
    bool enable_event_history{true};
    size_t max_event_history{1000};
    
    // Timing constraints
    std::chrono::microseconds max_event_processing_time{1000};  // 1ms
    std::chrono::microseconds event_timeout{10000};           // 10ms
    bool strict_timing{false};
    
    // Control loop settings
    std::chrono::microseconds control_loop_period{1000};      // 1ms
    bool enable_control_loop{false};
    double control_tolerance{1e-6};
    
    // Sensor settings
    std::chrono::microseconds sensor_timeout{5000};           // 5ms
    bool enable_sensor_validation{true};
    double sensor_noise_threshold{1e-3};
    
    // Threading
    size_t event_thread_pool_size{2};
    bool enable_async_processing{true};
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (max_event_history == 0) {
            throw std::invalid_argument("max_event_history must be positive");
        }
        
        if (max_event_processing_time <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("max_event_processing_time must be positive");
        }
        
        if (event_timeout <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("event_timeout must be positive");
        }
        
        if (control_loop_period <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("control_loop_period must be positive");
        }
        
        if (sensor_timeout <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("sensor_timeout must be positive");
        }
        
        if (control_tolerance <= 0) {
            throw std::invalid_argument("control_tolerance must be positive");
        }
        
        if (sensor_noise_threshold <= 0) {
            throw std::invalid_argument("sensor_noise_threshold must be positive");
        }
        
        if (event_thread_pool_size == 0) {
            throw std::invalid_argument("event_thread_pool_size must be positive");
        }
    }
};

/**
 * @brief Event data structure
 */
template<typename S, typename T>
struct Event {
    EventTrigger trigger;
    EventPriority priority;
    T timestamp;
    std::string event_id;
    std::function<void(S&, T)> handler;
    std::vector<uint8_t> data;
    
    // Metadata
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point processed_at;
    bool processed{false};
    bool timed_out{false};
    
    Event(EventTrigger t, EventPriority p, T time, std::string id, std::function<void(S&, T)> h)
        : trigger(t), priority(p), timestamp(time), event_id(std::move(id)), handler(std::move(h))
        , created_at(std::chrono::steady_clock::now()) {}
    
    // Comparison for priority queue (higher priority first)
    bool operator<(const Event& other) const {
        if (priority != other.priority) {
            return priority < other.priority;
        }
        return timestamp > other.timestamp;  // Earlier timestamp first for same priority
    }
};

/**
 * @brief Sensor data structure
 */
template<typename T>
struct SensorData {
    std::string sensor_id;
    std::vector<double> values;
    T timestamp;
    double confidence{1.0};
    bool valid{true};
    
    SensorData(std::string id, std::vector<double> vals, T time)
        : sensor_id(std::move(id)), values(std::move(vals)), timestamp(time) {}
};

/**
 * @brief Control feedback structure
 */
template<typename S, typename T>
struct ControlFeedback {
    std::string control_id;
    S target_state;
    S current_state;
    S error_state;
    T timestamp;
    double performance_metric{0.0};
    
    ControlFeedback(std::string id, S target, S current, T time)
        : control_id(std::move(id)), target_state(std::move(target)), current_state(std::move(current)), timestamp(time) {
        // Calculate error state
        error_state = target_state;
        for (size_t i = 0; i < error_state.size(); ++i) {
            error_state[i] = target_state[i] - current_state[i];
        }
    }
};

/**
 * @brief Event statistics
 */
struct EventStats {
    size_t total_events{0};
    size_t processed_events{0};
    size_t timed_out_events{0};
    size_t high_priority_events{0};
    size_t control_feedback_events{0};
    size_t sensor_events{0};
    std::chrono::microseconds total_processing_time{0};
    std::chrono::microseconds max_processing_time{0};
    std::chrono::microseconds min_processing_time{std::chrono::microseconds::max()};
    
    double average_processing_time_us() const {
        return processed_events > 0 ? 
            static_cast<double>(total_processing_time.count()) / processed_events : 0.0;
    }
    
    double event_success_rate() const {
        return total_events > 0 ? 
            static_cast<double>(processed_events) / total_events : 0.0;
    }
    
    double timeout_rate() const {
        return total_events > 0 ? 
            static_cast<double>(timed_out_events) / total_events : 0.0;
    }
};

/**
 * @brief Event decorator - adds event-driven feedback capabilities to any integrator
 * 
 * This decorator provides comprehensive event-driven capabilities with the following features:
 * - Multiple event trigger types (time, state, sensor, control)
 * - Priority-based event processing
 * - Real-time constraints and timing guarantees
 * - Sensor data integration and validation
 * - Control loop feedback mechanisms
 * - Asynchronous event processing
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles event-driven feedback
 * - Real-time: Designed for robotics and control applications
 * - Flexible: Multiple trigger types and processing modes
 * - Robust: Timeout handling and error recovery
 */
template<system_state S, can_be_time T = double>
class EventDecorator : public IntegratorDecorator<S, T> {
private:
    EventConfig config_;
    std::priority_queue<Event<S, T>> event_queue_;
    std::vector<Event<S, T>> event_history_;
    std::map<std::string, SensorData<T>> sensor_data_;
    std::map<std::string, ControlFeedback<S, T>> control_feedback_;
    EventStats stats_;
    
    // Threading for async processing
    std::vector<std::thread> event_threads_;
    std::atomic<bool> running_{false};
    std::mutex event_queue_mutex_;
    std::condition_variable event_queue_cv_;
    std::mutex sensor_data_mutex_;
    std::mutex control_feedback_mutex_;
    std::mutex stats_mutex_;
    
    // Control loop
    std::thread control_loop_thread_;
    std::atomic<bool> control_loop_running_{false};
    std::chrono::steady_clock::time_point last_control_update_;
    
    // Event callbacks
    std::map<EventTrigger, std::vector<std::function<void(S&, T)>>> event_callbacks_;

public:
    /**
     * @brief Construct event decorator
     * @param integrator The integrator to wrap
     * @param config Event configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit EventDecorator(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                           EventConfig config = {})
        : IntegratorDecorator<S, T>(std::move(integrator)), config_(std::move(config))
        , last_control_update_(std::chrono::steady_clock::now()) {
        
        config_.validate();
        
        if (config_.enable_async_processing) {
            start_event_processing();
        }
        
        if (config_.enable_control_loop) {
            start_control_loop();
        }
    }
    
    /**
     * @brief Destructor ensures proper cleanup
     */
    ~EventDecorator() {
        stop_control_loop();
        stop_event_processing();
    }

    /**
     * @brief Override step to handle events during integration
     */
    void step(typename IntegratorDecorator<S, T>::state_type& state, T dt) override {
        // Process pending events before step
        process_events(state);
        
        // Perform integration step
        this->wrapped_integrator_->step(state, dt);
        
        // Check for state-based events after step
        check_state_events(state, this->current_time());
        
        // Process any new events
        process_events(state);
    }

    /**
     * @brief Override integrate to handle events during integration
     */
    void integrate(typename IntegratorDecorator<S, T>::state_type& state, T dt, T end_time) override {
        // Process initial events
        process_events(state);
        
        // Integrate with event handling
        this->wrapped_integrator_->integrate(state, dt, end_time);
        
        // Final event processing
        process_events(state);
    }

    /**
     * @brief Register event handler for specific trigger type
     * @param trigger Event trigger type
     * @param handler Event handler function
     */
    void register_event_handler(EventTrigger trigger, std::function<void(S&, T)> handler) {
        event_callbacks_[trigger].push_back(std::move(handler));
    }

    /**
     * @brief Trigger custom event
     * @param event_id Unique event identifier
     * @param priority Event priority
     * @param handler Event handler function
     * @param data Optional event data
     */
    void trigger_event(const std::string& event_id, EventPriority priority, 
                      std::function<void(S&, T)> handler, std::vector<uint8_t> data = {}) {
        std::lock_guard<std::mutex> lock(event_queue_mutex_);
        
        Event<S, T> event(EventTrigger::CUSTOM, priority, this->current_time(), event_id, std::move(handler));
        event.data = std::move(data);
        
        event_queue_.push(event);
        event_queue_cv_.notify_one();
        
        stats_.total_events++;
        if (priority >= EventPriority::HIGH) {
            stats_.high_priority_events++;
        }
    }

    /**
     * @brief Submit sensor data
     * @param sensor_id Sensor identifier
     * @param values Sensor values
     * @param confidence Confidence level (0.0 to 1.0)
     */
    void submit_sensor_data(const std::string& sensor_id, const std::vector<double>& values, double confidence = 1.0) {
        std::lock_guard<std::mutex> lock(sensor_data_mutex_);
        
        SensorData<T> sensor_data(sensor_id, values, this->current_time());
        sensor_data.confidence = confidence;
        sensor_data.valid = validate_sensor_data(sensor_data);
        
        sensor_data_[sensor_id] = sensor_data;
        
        // Trigger sensor event
        trigger_event("sensor_" + sensor_id, EventPriority::NORMAL, 
                     [this, sensor_id](S& state, T time) {
                         handle_sensor_event(sensor_id, state, time);
                     });
        
        stats_.sensor_events++;
    }

    /**
     * @brief Submit control feedback
     * @param control_id Control identifier
     * @param target_state Target state
     * @param current_state Current state
     */
    void submit_control_feedback(const std::string& control_id, const S& target_state, const S& current_state) {
        std::lock_guard<std::mutex> lock(control_feedback_mutex_);
        
        ControlFeedback<S, T> feedback(control_id, target_state, current_state, this->current_time());
        control_feedback_[control_id] = feedback;
        
        // Trigger control feedback event
        trigger_event("control_" + control_id, EventPriority::HIGH,
                     [this, control_id](S& state, T time) {
                         handle_control_feedback_event(control_id, state, time);
                     });
        
        stats_.control_feedback_events++;
    }

    /**
     * @brief Set state-based event condition
     * @param condition Function that returns true when event should trigger
     * @param handler Event handler function
     * @param priority Event priority
     */
    void set_state_condition(std::function<bool(const S&, T)> condition, 
                            std::function<void(S&, T)> handler, 
                            EventPriority priority = EventPriority::NORMAL) {
        register_event_handler(EventTrigger::STATE_BASED, 
                              [condition, handler, priority, this](S& state, T time) {
                                  if (condition(state, time)) {
                                      handler(state, time);
                                  }
                              });
    }

    /**
     * @brief Set threshold crossing event
     * @param state_index Index of state variable to monitor
     * @param threshold Threshold value
     * @param crossing_direction true for upward crossing, false for downward
     * @param handler Event handler function
     */
    void set_threshold_event(size_t state_index, double threshold, bool crossing_direction,
                            std::function<void(S&, T)> handler) {
        static std::map<size_t, double> last_values;
        
        register_event_handler(EventTrigger::THRESHOLD_CROSSING,
                              [state_index, threshold, crossing_direction, handler, this](S& state, T time) {
                                  if (state_index >= state.size()) return;
                                  
                                  double current_value = state[state_index];
                                  double last_value = last_values[state_index];
                                  
                                  bool crossed = crossing_direction ? 
                                      (last_value < threshold && current_value >= threshold) :
                                      (last_value > threshold && current_value <= threshold);
                                  
                                  if (crossed) {
                                      handler(state, time);
                                  }
                                  
                                  last_values[state_index] = current_value;
                              });
    }

    /**
     * @brief Get event statistics
     */
    const EventStats& get_statistics() const {
        return stats_;
    }

    /**
     * @brief Reset event statistics
     */
    void reset_statistics() {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_ = EventStats{};
    }

    /**
     * @brief Get event history
     */
    const std::vector<Event<S, T>>& get_event_history() const {
        return event_history_;
    }

    /**
     * @brief Clear event history
     */
    void clear_event_history() {
        event_history_.clear();
    }

    /**
     * @brief Get current sensor data
     */
    std::map<std::string, SensorData<T>> get_sensor_data() const {
        std::lock_guard<std::mutex> lock(sensor_data_mutex_);
        return sensor_data_;
    }

    /**
     * @brief Get current control feedback
     */
    std::map<std::string, ControlFeedback<S, T>> get_control_feedback() const {
        std::lock_guard<std::mutex> lock(control_feedback_mutex_);
        return control_feedback_;
    }

    /**
     * @brief Access and modify event configuration
     */
    EventConfig& config() { return config_; }
    const EventConfig& config() const { return config_; }

private:
    /**
     * @brief Process events from queue
     */
    void process_events(S& state) {
        auto deadline = std::chrono::steady_clock::now() + config_.max_event_processing_time;
        
        while (!event_queue_.empty() && std::chrono::steady_clock::now() < deadline) {
            std::lock_guard<std::mutex> lock(event_queue_mutex_);
            
            if (event_queue_.empty()) break;
            
            Event<S, T> event = event_queue_.top();
            event_queue_.pop();
            
            process_single_event(event, state);
        }
    }

    /**
     * @brief Process a single event
     */
    void process_single_event(Event<S, T>& event, S& state) {
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            if (event.handler) {
                event.handler(state, event.timestamp);
            }
            
            event.processed = true;
            event.processed_at = std::chrono::steady_clock::now();
            
            auto processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                event.processed_at - start_time);
            
            // Update statistics
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.processed_events++;
            stats_.total_processing_time += processing_time;
            stats_.max_processing_time = std::max(stats_.max_processing_time, processing_time);
            stats_.min_processing_time = std::min(stats_.min_processing_time, processing_time);
            
        } catch (const std::exception& e) {
            // Log error but don't stop processing
            event.timed_out = true;
            stats_.timed_out_events++;
        }
        
        // Add to history if enabled
        if (config_.enable_event_history) {
            event_history_.push_back(event);
            if (event_history_.size() > config_.max_event_history) {
                event_history_.erase(event_history_.begin());
            }
        }
    }

    /**
     * @brief Check for state-based events
     */
    void check_state_events(const S& state, T time) {
        for (const auto& handler : event_callbacks_[EventTrigger::STATE_BASED]) {
            // Create a copy of the state for the handler
            S state_copy = state;
            handler(state_copy, time);
        }
    }

    /**
     * @brief Handle sensor event
     */
    void handle_sensor_event(const std::string& sensor_id, S& state, T time) {
        std::lock_guard<std::mutex> lock(sensor_data_mutex_);
        
        auto it = sensor_data_.find(sensor_id);
        if (it != sensor_data_.end() && it->second.valid) {
            // Process sensor data - this is application-specific
            // For example, update state based on sensor feedback
            
            // Call registered sensor callbacks
            for (const auto& handler : event_callbacks_[EventTrigger::SENSOR_DATA]) {
                handler(state, time);
            }
        }
    }

    /**
     * @brief Handle control feedback event
     */
    void handle_control_feedback_event(const std::string& control_id, S& state, T time) {
        std::lock_guard<std::mutex> lock(control_feedback_mutex_);
        
        auto it = control_feedback_.find(control_id);
        if (it != control_feedback_.end()) {
            const auto& feedback = it->second;
            
            // Apply control correction based on error
            for (size_t i = 0; i < state.size() && i < feedback.error_state.size(); ++i) {
                state[i] += feedback.error_state[i] * 0.1;  // Simple proportional control
            }
            
            // Call registered control callbacks
            for (const auto& handler : event_callbacks_[EventTrigger::CONTROL_FEEDBACK]) {
                handler(state, time);
            }
        }
    }

    /**
     * @brief Validate sensor data
     */
    bool validate_sensor_data(const SensorData<T>& sensor_data) {
        if (!config_.enable_sensor_validation) {
            return true;
        }
        
        // Check for reasonable values
        for (double value : sensor_data.values) {
            if (std::isnan(value) || std::isinf(value)) {
                return false;
            }
            
            // Check noise threshold
            if (std::abs(value) < config_.sensor_noise_threshold) {
                // Might be noise, but still valid
            }
        }
        
        return sensor_data.confidence > 0.1;  // Minimum confidence threshold
    }

    /**
     * @brief Start event processing threads
     */
    void start_event_processing() {
        running_ = true;
        
        for (size_t i = 0; i < config_.event_thread_pool_size; ++i) {
            event_threads_.emplace_back([this]() {
                while (running_) {
                    std::unique_lock<std::mutex> lock(event_queue_mutex_);
                    
                    if (event_queue_cv_.wait_for(lock, std::chrono::milliseconds(100), 
                                                [this] { return !event_queue_.empty() || !running_; })) {
                        
                        if (!running_) break;
                        
                        if (!event_queue_.empty()) {
                            Event<S, T> event = event_queue_.top();
                            event_queue_.pop();
                            lock.unlock();
                            
                            // Process event asynchronously
                            // Note: This would need access to current state, which is tricky
                            // In practice, async events might be limited to specific types
                        }
                    }
                }
            });
        }
    }

    /**
     * @brief Stop event processing threads
     */
    void stop_event_processing() {
        running_ = false;
        event_queue_cv_.notify_all();
        
        for (auto& thread : event_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        event_threads_.clear();
    }

    /**
     * @brief Start control loop thread
     */
    void start_control_loop() {
        control_loop_running_ = true;
        
        control_loop_thread_ = std::thread([this]() {
            while (control_loop_running_) {
                auto now = std::chrono::steady_clock::now();
                
                if (now - last_control_update_ >= config_.control_loop_period) {
                    // Trigger control loop events
                    for (const auto& handler : event_callbacks_[EventTrigger::CONTROL_FEEDBACK]) {
                        // Note: This needs access to current state
                        // Would need to be implemented differently in practice
                    }
                    
                    last_control_update_ = now;
                }
                
                std::this_thread::sleep_for(config_.control_loop_period / 10);
            }
        });
    }

    /**
     * @brief Stop control loop thread
     */
    void stop_control_loop() {
        control_loop_running_ = false;
        
        if (control_loop_thread_.joinable()) {
            control_loop_thread_.join();
        }
    }
};

} // namespace diffeq::core::composable 