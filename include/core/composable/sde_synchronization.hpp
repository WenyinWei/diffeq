#pragma once

#include "interprocess_decorator.hpp"
#include <functional>
#include <memory>
#include <chrono>
#include <random>

namespace diffeq::core::composable {

/**
 * @brief SDE synchronization mode
 */
enum class SDESyncMode {
    IMMEDIATE,          // Noise data required immediately
    BUFFERED,           // Buffer noise data for smooth delivery
    INTERPOLATED,       // Interpolate between noise samples
    GENERATED           // Generate noise locally with synchronized seed
};

/**
 * @brief Noise process type
 */
enum class NoiseProcessType {
    WIENER,             // Standard Wiener process (Brownian motion)
    COLORED_NOISE,      // Colored noise with correlation
    JUMP_PROCESS,       // Jump diffusion process
    LEVY_PROCESS,       // Lévy process
    CUSTOM              // Custom noise process
};

/**
 * @brief Configuration for SDE synchronization
 */
struct SDESyncConfig {
    SDESyncMode sync_mode{SDESyncMode::BUFFERED};
    NoiseProcessType noise_type{NoiseProcessType::WIENER};
    
    // Timing parameters
    std::chrono::microseconds max_noise_delay{1000};      // 1ms
    std::chrono::microseconds noise_buffer_time{5000};    // 5ms
    std::chrono::microseconds sync_timeout{10000};        // 10ms
    
    // Noise generation parameters
    uint64_t random_seed{12345};
    double noise_intensity{1.0};
    size_t noise_dimensions{1};
    
    // Buffering and interpolation
    size_t buffer_size{1000};
    bool enable_interpolation{true};
    double interpolation_tolerance{1e-6};
    
    // Network synchronization
    bool enable_time_sync{true};
    std::chrono::microseconds time_sync_interval{1000};   // 1ms
    
    /**
     * @brief Validate configuration
     */
    void validate() const {
        if (max_noise_delay <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("max_noise_delay must be positive");
        }
        
        if (noise_buffer_time <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("noise_buffer_time must be positive");
        }
        
        if (sync_timeout <= std::chrono::microseconds{0}) {
            throw std::invalid_argument("sync_timeout must be positive");
        }
        
        if (noise_dimensions == 0) {
            throw std::invalid_argument("noise_dimensions must be positive");
        }
        
        if (buffer_size == 0) {
            throw std::invalid_argument("buffer_size must be positive");
        }
        
        if (interpolation_tolerance <= 0) {
            throw std::invalid_argument("interpolation_tolerance must be positive");
        }
    }
};

/**
 * @brief Noise data structure
 */
template<typename T>
struct NoiseData {
    T timestamp;
    std::vector<double> increments;
    NoiseProcessType type;
    uint64_t sequence_number{0};
    
    NoiseData(T time, std::vector<double> inc, NoiseProcessType t)
        : timestamp(time), increments(std::move(inc)), type(t) {}
};

/**
 * @brief Noise process generator interface
 */
template<typename T>
class NoiseProcessGenerator {
public:
    virtual ~NoiseProcessGenerator() = default;
    
    virtual NoiseData<T> generate_increment(T current_time, T dt) = 0;
    virtual void reset_seed(uint64_t seed) = 0;
    virtual std::string get_process_name() const = 0;
};

/**
 * @brief Standard Wiener process generator
 */
template<typename T>
class WienerProcessGenerator : public NoiseProcessGenerator<T> {
private:
    std::mt19937_64 rng_;
    std::normal_distribution<double> normal_dist_;
    size_t dimensions_;
    double intensity_;

public:
    explicit WienerProcessGenerator(size_t dimensions, double intensity = 1.0, uint64_t seed = 12345)
        : rng_(seed), normal_dist_(0.0, 1.0), dimensions_(dimensions), intensity_(intensity) {}
    
    NoiseData<T> generate_increment(T current_time, T dt) override {
        std::vector<double> increments;
        increments.reserve(dimensions_);
        
        // Generate independent Gaussian increments
        double sqrt_dt = std::sqrt(static_cast<double>(dt));
        for (size_t i = 0; i < dimensions_; ++i) {
            increments.push_back(intensity_ * normal_dist_(rng_) * sqrt_dt);
        }
        
        return NoiseData<T>(current_time, std::move(increments), NoiseProcessType::WIENER);
    }
    
    void reset_seed(uint64_t seed) override {
        rng_.seed(seed);
    }
    
    std::string get_process_name() const override {
        return "Wiener Process (Brownian Motion)";
    }
};

/**
 * @brief SDE synchronization helper for coordinating noise processes
 * 
 * This class provides utilities for synchronizing SDE integrators that need
 * to receive noise data from external processes. It handles:
 * - Noise data buffering and interpolation
 * - Time synchronization between processes
 * - Multiple noise process types
 * - Robust error handling and recovery
 */
template<system_state S, can_be_time T = double>
class SDESynchronizer {
private:
    SDESyncConfig config_;
    std::unique_ptr<NoiseProcessGenerator<T>> local_generator_;
    std::vector<NoiseData<T>> noise_buffer_;
    std::mutex noise_mutex_;
    std::condition_variable noise_cv_;
    std::atomic<bool> noise_available_{false};
    
    // Time synchronization
    std::chrono::steady_clock::time_point reference_time_;
    std::atomic<T> synchronized_time_{};
    
    // Statistics
    size_t noise_requests_{0};
    size_t noise_timeouts_{0};
    size_t interpolations_{0};

public:
    /**
     * @brief Construct SDE synchronizer
     * @param config SDE synchronization configuration
     */
    explicit SDESynchronizer(SDESyncConfig config = {})
        : config_(std::move(config)), reference_time_(std::chrono::steady_clock::now()) {
        
        config_.validate();
        initialize_local_generator();
    }

    /**
     * @brief Get noise increment for SDE integration
     * @param current_time Current integration time
     * @param dt Time step
     * @return Noise data for the time step
     * @throws std::runtime_error if noise cannot be obtained within timeout
     */
    NoiseData<T> get_noise_increment(T current_time, T dt) {
        noise_requests_++;
        
        switch (config_.sync_mode) {
            case SDESyncMode::IMMEDIATE:
                return get_immediate_noise(current_time, dt);
            case SDESyncMode::BUFFERED:
                return get_buffered_noise(current_time, dt);
            case SDESyncMode::INTERPOLATED:
                return get_interpolated_noise(current_time, dt);
            case SDESyncMode::GENERATED:
                return get_generated_noise(current_time, dt);
            default:
                throw std::runtime_error("Unknown SDE synchronization mode");
        }
    }

    /**
     * @brief Submit noise data from external process
     * @param noise_data Noise data received
     */
    void submit_noise_data(const NoiseData<T>& noise_data) {
        std::lock_guard<std::mutex> lock(noise_mutex_);
        
        // Insert in chronological order
        auto it = std::lower_bound(noise_buffer_.begin(), noise_buffer_.end(), noise_data,
                                  [](const NoiseData<T>& a, const NoiseData<T>& b) {
                                      return a.timestamp < b.timestamp;
                                  });
        
        noise_buffer_.insert(it, noise_data);
        
        // Limit buffer size
        if (noise_buffer_.size() > config_.buffer_size) {
            noise_buffer_.erase(noise_buffer_.begin());
        }
        
        noise_available_ = true;
        noise_cv_.notify_all();
    }

    /**
     * @brief Configure for use with InterprocessDecorator
     * @param ipc_decorator Interprocess decorator to coordinate with
     */
    template<typename IPCDecorator>
    void configure_with_ipc(IPCDecorator& ipc_decorator) {
        // Set up callback to receive noise data
        ipc_decorator.set_receive_callback([this](const S& state, T time) {
            // Deserialize noise data from state
            // This is simplified - real implementation would need proper serialization
            if (state.size() >= config_.noise_dimensions) {
                std::vector<double> increments(state.begin(), state.begin() + config_.noise_dimensions);
                NoiseData<T> noise_data(time, std::move(increments), config_.noise_type);
                submit_noise_data(noise_data);
            }
        });
    }

    /**
     * @brief Create synchronized SDE integrator pair
     * @param producer_integrator Integrator that generates noise
     * @param consumer_integrator Integrator that receives noise
     * @param ipc_config IPC configuration
     * @return Pair of configured integrators
     */
    template<typename ProducerIntegrator, typename ConsumerIntegrator>
    static std::pair<std::unique_ptr<AbstractIntegrator<S, T>>, std::unique_ptr<AbstractIntegrator<S, T>>>
    create_synchronized_pair(std::unique_ptr<ProducerIntegrator> producer_integrator,
                            std::unique_ptr<ConsumerIntegrator> consumer_integrator,
                            InterprocessConfig ipc_config,
                            SDESyncConfig sync_config = {}) {
        
        // Configure producer (noise generator)
        InterprocessConfig producer_config = ipc_config;
        producer_config.direction = IPCDirection::PRODUCER;
        
        auto producer = make_builder(std::move(producer_integrator))
            .with_interprocess(producer_config)
            .build();
        
        // Configure consumer (noise receiver)
        InterprocessConfig consumer_config = ipc_config;
        consumer_config.direction = IPCDirection::CONSUMER;
        consumer_config.sync_mode = IPCSyncMode::BLOCKING;  // SDE needs synchronous noise
        
        auto consumer = make_builder(std::move(consumer_integrator))
            .with_interprocess(consumer_config)
            .build();
        
        return {std::move(producer), std::move(consumer)};
    }

    /**
     * @brief Get synchronization statistics
     */
    struct SyncStats {
        size_t noise_requests;
        size_t noise_timeouts;
        size_t interpolations;
        double timeout_rate() const {
            return noise_requests > 0 ? static_cast<double>(noise_timeouts) / noise_requests : 0.0;
        }
    };
    
    SyncStats get_statistics() const {
        return {noise_requests_, noise_timeouts_, interpolations_};
    }

    /**
     * @brief Reset statistics
     */
    void reset_statistics() {
        noise_requests_ = 0;
        noise_timeouts_ = 0;
        interpolations_ = 0;
    }

private:
    /**
     * @brief Initialize local noise generator
     */
    void initialize_local_generator() {
        switch (config_.noise_type) {
            case NoiseProcessType::WIENER:
                local_generator_ = std::make_unique<WienerProcessGenerator<T>>(
                    config_.noise_dimensions, config_.noise_intensity, config_.random_seed);
                break;
            default:
                local_generator_ = std::make_unique<WienerProcessGenerator<T>>(
                    config_.noise_dimensions, config_.noise_intensity, config_.random_seed);
                break;
        }
    }

    /**
     * @brief Get immediate noise (blocking)
     */
    NoiseData<T> get_immediate_noise(T current_time, T dt) {
        std::unique_lock<std::mutex> lock(noise_mutex_);
        
        if (noise_cv_.wait_for(lock, config_.sync_timeout, [this] { return noise_available_.load(); })) {
            // Find noise data at or near current time
            auto it = std::lower_bound(noise_buffer_.begin(), noise_buffer_.end(), current_time,
                                      [](const NoiseData<T>& noise, T time) {
                                          return noise.timestamp < time;
                                      });
            
            if (it != noise_buffer_.end()) {
                NoiseData<T> result = *it;
                noise_buffer_.erase(it);
                noise_available_ = !noise_buffer_.empty();
                return result;
            }
        }
        
        // Timeout - generate locally or throw
        noise_timeouts_++;
        return local_generator_->generate_increment(current_time, dt);
    }

    /**
     * @brief Get buffered noise
     */
    NoiseData<T> get_buffered_noise(T current_time, T dt) {
        std::lock_guard<std::mutex> lock(noise_mutex_);
        
        // Look for buffered noise near current time
        for (auto it = noise_buffer_.begin(); it != noise_buffer_.end(); ++it) {
            if (std::abs(it->timestamp - current_time) < static_cast<T>(config_.max_noise_delay.count()) / 1e6) {
                NoiseData<T> result = *it;
                noise_buffer_.erase(it);
                return result;
            }
        }
        
        // No suitable buffered noise - generate locally
        return local_generator_->generate_increment(current_time, dt);
    }

    /**
     * @brief Get interpolated noise
     */
    NoiseData<T> get_interpolated_noise(T current_time, T dt) {
        std::lock_guard<std::mutex> lock(noise_mutex_);
        
        if (noise_buffer_.size() < 2) {
            return local_generator_->generate_increment(current_time, dt);
        }
        
        // Find surrounding noise data points
        auto upper = std::lower_bound(noise_buffer_.begin(), noise_buffer_.end(), current_time,
                                     [](const NoiseData<T>& noise, T time) {
                                         return noise.timestamp < time;
                                     });
        
        if (upper == noise_buffer_.begin() || upper == noise_buffer_.end()) {
            return local_generator_->generate_increment(current_time, dt);
        }
        
        auto lower = std::prev(upper);
        
        // Linear interpolation
        T alpha = (current_time - lower->timestamp) / (upper->timestamp - lower->timestamp);
        
        std::vector<double> interpolated_increments;
        interpolated_increments.reserve(config_.noise_dimensions);
        
        for (size_t i = 0; i < config_.noise_dimensions && i < lower->increments.size() && i < upper->increments.size(); ++i) {
            double interpolated = (1 - alpha) * lower->increments[i] + alpha * upper->increments[i];
            interpolated_increments.push_back(interpolated);
        }
        
        interpolations_++;
        return NoiseData<T>(current_time, std::move(interpolated_increments), config_.noise_type);
    }

    /**
     * @brief Get locally generated noise
     */
    NoiseData<T> get_generated_noise(T current_time, T dt) {
        return local_generator_->generate_increment(current_time, dt);
    }
};

// ============================================================================
// CONVENIENCE FUNCTIONS FOR SDE SYNCHRONIZATION
// ============================================================================

/**
 * @brief Create Wiener process generator integrator
 * @param dimensions Number of noise dimensions
 * @param intensity Noise intensity
 * @param channel_name IPC channel name
 * @return Configured noise generator integrator
 */
template<system_state S, can_be_time T = double>
auto create_wiener_process_generator(size_t dimensions, double intensity = 1.0, const std::string& channel_name = "wiener_process") {
    // This would be a specialized integrator that generates Wiener process increments
    // and sends them via IPC. Implementation would depend on specific requirements.
    
    InterprocessConfig ipc_config;
    ipc_config.direction = IPCDirection::PRODUCER;
    ipc_config.channel_name = channel_name;
    
    SDESyncConfig sync_config;
    sync_config.noise_type = NoiseProcessType::WIENER;
    sync_config.noise_dimensions = dimensions;
    sync_config.noise_intensity = intensity;
    
    // Note: This is a placeholder - would need actual Wiener process integrator implementation
    return std::make_pair(ipc_config, sync_config);
}

/**
 * @brief Configure SDE integrator to receive external noise
 * @param integrator SDE integrator to configure
 * @param channel_name IPC channel name for receiving noise
 * @return Configured integrator with noise synchronization
 */
template<typename SDEIntegrator>
auto configure_for_external_noise(std::unique_ptr<SDEIntegrator> integrator, const std::string& channel_name = "wiener_process") {
    InterprocessConfig ipc_config;
    ipc_config.direction = IPCDirection::CONSUMER;
    ipc_config.channel_name = channel_name;
    ipc_config.sync_mode = IPCSyncMode::BLOCKING;  // SDE requires synchronous noise
    
    return make_builder(std::move(integrator))
        .with_interprocess(ipc_config)
        .build();
}

} // namespace diffeq::core::composable
 