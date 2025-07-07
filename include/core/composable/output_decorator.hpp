#pragma once

#include "integrator_decorator.hpp"
#include <vector>
#include <functional>
#include <chrono>
#include <fstream>
#include <sstream>

namespace diffeq::core::composable {

/**
 * @brief Output mode enumeration
 */
enum class OutputMode {
    ONLINE,     // Real-time output during integration
    OFFLINE,    // Buffered output after integration
    HYBRID      // Combination of online and offline
};

/**
 * @brief Configuration for output handling
 */
struct OutputConfig {
    OutputMode mode{OutputMode::ONLINE};
    std::chrono::microseconds output_interval{1000};
    size_t buffer_size{1000};
    bool enable_compression{false};
    bool enable_file_output{false};
    std::string output_filename;
    bool append_to_file{false};
    
    // Validation settings
    bool validate_intervals{true};
    std::chrono::microseconds min_output_interval{10};  // Minimum 10μs
    std::chrono::microseconds max_output_interval{std::chrono::minutes{1}};  // Maximum 1 minute
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (validate_intervals) {
            if (output_interval < min_output_interval) {
                throw std::invalid_argument("output_interval below minimum " + 
                                          std::to_string(min_output_interval.count()) + "μs");
            }
            if (output_interval > max_output_interval) {
                throw std::invalid_argument("output_interval exceeds maximum " + 
                                          std::to_string(max_output_interval.count()) + "μs");
            }
        }
        
        if (buffer_size == 0) {
            throw std::invalid_argument("buffer_size must be positive");
        }
        
        if (enable_file_output && output_filename.empty()) {
            throw std::invalid_argument("output_filename required when file output is enabled");
        }
    }
};

/**
 * @brief Output statistics and information
 */
struct OutputStats {
    size_t total_outputs{0};
    size_t online_outputs{0};
    size_t buffered_outputs{0};
    size_t file_writes{0};
    std::chrono::milliseconds total_output_time{0};
    size_t buffer_overflows{0};
    
    double average_output_time_ms() const {
        return total_outputs > 0 ? 
            static_cast<double>(total_output_time.count()) / total_outputs : 0.0;
    }
};

/**
 * @brief Output decorator - adds configurable output to any integrator
 * 
 * This decorator provides comprehensive output capabilities with the following features:
 * - Online, offline, and hybrid output modes
 * - Configurable output intervals and buffering
 * - Optional file output with compression
 * - Detailed statistics and performance monitoring
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles output functionality
 * - No Dependencies: Works with any integrator type
 * - Flexible: Multiple output modes and configurations
 * - Efficient: Minimal performance impact on integration
 */
template<system_state S>
class OutputDecorator : public IntegratorDecorator<S> {
private:
    OutputConfig config_;
    std::function<void(const S&, typename IntegratorDecorator<S>::time_type, size_t)> output_handler_;
    std::vector<std::tuple<S, typename IntegratorDecorator<S>::time_type, size_t>> output_buffer_;
    std::chrono::steady_clock::time_point last_output_;
    size_t step_count_{0};
    OutputStats stats_;
    std::unique_ptr<std::ofstream> output_file_;

public:
    /**
     * @brief Construct output decorator
     * @param integrator The integrator to wrap
     * @param config Output configuration (validated on construction)
     * @param handler Optional output handler function
     * @throws std::invalid_argument if config is invalid
     */
    explicit OutputDecorator(std::unique_ptr<AbstractIntegrator<S>> integrator,
                            OutputConfig config = {},
                            std::function<void(const S&, typename IntegratorDecorator<S>::time_type, size_t)> handler = nullptr)
        : IntegratorDecorator<S>(std::move(integrator))
        , config_(std::move(config))
        , output_handler_(std::move(handler))
        , last_output_(std::chrono::steady_clock::now()) {
        
        config_.validate();
        initialize_file_output();
    }

    /**
     * @brief Destructor ensures proper cleanup and final output flush
     */
    ~OutputDecorator() {
        try {
            flush_output();
            if (output_file_ && output_file_->is_open()) {
                output_file_->close();
            }
        } catch (...) {
            // Swallow exceptions in destructor
        }
    }

    /**
     * @brief Override step to add output handling
     */
    void step(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt) override {
        this->wrapped_integrator_->step(state, dt);
        ++step_count_;
        
        handle_output(state, this->current_time());
    }

    /**
     * @brief Override integrate to handle different output modes
     */
    void integrate(typename IntegratorDecorator<S>::state_type& state, typename IntegratorDecorator<S>::time_type dt, 
                   typename IntegratorDecorator<S>::time_type end_time) override {
        if (config_.mode == OutputMode::OFFLINE) {
            // Just integrate and buffer final result
            this->wrapped_integrator_->integrate(state, dt, end_time);
            buffer_output(state, this->current_time(), step_count_);
        } else {
            // Step-by-step with online output
            while (this->current_time() < end_time) {
                typename IntegratorDecorator<S>::time_type step_size = std::min(dt, end_time - this->current_time());
                this->step(state, step_size);
            }
        }
        
        if (config_.mode == OutputMode::OFFLINE || config_.mode == OutputMode::HYBRID) {
            flush_output();
        }
    }

    /**
     * @brief Set or change output handler function
     * @param handler New output handler function
     */
    void set_output_handler(std::function<void(const S&, typename IntegratorDecorator<S>::time_type, size_t)> handler) {
        output_handler_ = std::move(handler);
    }

    /**
     * @brief Get current output buffer contents
     * @return Reference to the output buffer
     */
    const std::vector<std::tuple<S, typename IntegratorDecorator<S>::time_type, size_t>>& get_buffer() const { 
        return output_buffer_; 
    }
    
    /**
     * @brief Clear the output buffer
     */
    void clear_buffer() { 
        output_buffer_.clear();
        stats_.buffered_outputs = 0;
    }

    /**
     * @brief Force immediate output flush
     */
    void flush_output() {
        if (output_handler_) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (const auto& [state, time, step] : output_buffer_) {
                output_handler_(state, time, step);
                stats_.total_outputs++;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            stats_.total_output_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);
        }
        
        if (config_.enable_file_output && output_file_ && output_file_->is_open()) {
            write_buffer_to_file();
        }
    }

    /**
     * @brief Get output statistics
     */
    const OutputStats& get_statistics() const {
        return stats_;
    }

    /**
     * @brief Reset output statistics
     */
    void reset_statistics() {
        stats_ = OutputStats{};
    }

    /**
     * @brief Access and modify output configuration
     */
    OutputConfig& config() { return config_; }
    const OutputConfig& config() const { return config_; }
    
    /**
     * @brief Update output configuration with validation
     * @param new_config New configuration
     * @throws std::invalid_argument if new config is invalid
     */
    void update_config(OutputConfig new_config) {
        new_config.validate();
        
        // Check if file output settings changed
        bool file_settings_changed = (new_config.enable_file_output != config_.enable_file_output) ||
                                     (new_config.output_filename != config_.output_filename) ||
                                     (new_config.append_to_file != config_.append_to_file);
        
        config_ = std::move(new_config);
        
        if (file_settings_changed) {
            initialize_file_output();
        }
    }

private:
    /**
     * @brief Handle output based on current mode and configuration
     */
    void handle_output(const S& state, typename IntegratorDecorator<S>::time_type time) {
        auto now = std::chrono::steady_clock::now();
        
        if (config_.mode == OutputMode::ONLINE || config_.mode == OutputMode::HYBRID) {
            if (now - last_output_ >= config_.output_interval) {
                if (output_handler_) {
                    auto start_time = std::chrono::high_resolution_clock::now();
                    output_handler_(state, time, step_count_);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    
                    stats_.total_output_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time);
                    stats_.total_outputs++;
                    stats_.online_outputs++;
                }
                last_output_ = now;
            }
        }
        
        if (config_.mode == OutputMode::OFFLINE || config_.mode == OutputMode::HYBRID) {
            buffer_output(state, time, step_count_);
        }
    }

    /**
     * @brief Add data to output buffer
     */
    void buffer_output(const S& state, typename IntegratorDecorator<S>::time_type time, size_t step) {
        if (output_buffer_.size() >= config_.buffer_size) {
            output_buffer_.erase(output_buffer_.begin());
            stats_.buffer_overflows++;
        }
        output_buffer_.emplace_back(state, time, step);
        stats_.buffered_outputs++;
    }

    /**
     * @brief Initialize file output if enabled
     */
    void initialize_file_output() {
        if (config_.enable_file_output && !config_.output_filename.empty()) {
            output_file_ = std::make_unique<std::ofstream>();
            
            std::ios_base::openmode mode = std::ios_base::out;
            if (config_.append_to_file) {
                mode |= std::ios_base::app;
            }
            
            output_file_->open(config_.output_filename, mode);
            
            if (!output_file_->is_open()) {
                throw std::runtime_error("Failed to open output file: " + config_.output_filename);
            }
            
            // Write header if creating new file
            if (!config_.append_to_file) {
                *output_file_ << "# DiffEq Integration Output\n";
                *output_file_ << "# Time, State, Step\n";
            }
        }
    }

    /**
     * @brief Write buffer contents to file
     */
    void write_buffer_to_file() {
        if (!output_file_ || !output_file_->is_open()) {
            return;
        }
        
        for (const auto& [state, time, step] : output_buffer_) {
            *output_file_ << time << ", ";
            
            // Write state (assuming it's iterable)
            bool first = true;
            for (const auto& component : state) {
                if (!first) *output_file_ << " ";
                *output_file_ << component;
                first = false;
            }
            
            *output_file_ << ", " << step << "\n";
            stats_.file_writes++;
        }
        
        output_file_->flush();
    }
};

} // namespace diffeq::core::composable