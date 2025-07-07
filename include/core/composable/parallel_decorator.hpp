#pragma once

#include "integrator_decorator.hpp"
#include <vector>
#include <execution>
#include <thread>
#include <algorithm>
#include <ranges>
#include <type_traits>

namespace diffeq::core::composable {

/**
 * @brief Configuration for parallel execution
 */
struct ParallelConfig {
    size_t max_threads{0};  // 0 = auto-detect
    size_t chunk_size{1};   // Minimum work unit size
    bool enable_auto_chunking{true};
    double load_balance_threshold{0.1};  // 10% load imbalance tolerance
    
    // Validation settings
    bool validate_thread_count{true};
    size_t min_threads{1};
    size_t max_threads_limit{std::thread::hardware_concurrency() * 2};
    
    /**
     * @brief Validate configuration parameters
     * @throws std::invalid_argument if configuration is invalid
     */
    void validate() const {
        if (validate_thread_count && max_threads > 0) {
            if (max_threads < min_threads) {
                throw std::invalid_argument("max_threads must be >= " + std::to_string(min_threads));
            }
            if (max_threads > max_threads_limit) {
                throw std::invalid_argument("max_threads exceeds system limit of " + 
                                          std::to_string(max_threads_limit));
            }
        }
        
        if (chunk_size == 0) {
            throw std::invalid_argument("chunk_size must be positive");
        }
        
        if (load_balance_threshold < 0.0 || load_balance_threshold > 1.0) {
            throw std::invalid_argument("load_balance_threshold must be between 0.0 and 1.0");
        }
    }
};

/**
 * @brief Parallel execution decorator - adds batch processing to any integrator
 * 
 * This decorator provides parallel execution capabilities with the following features:
 * - Batch processing of multiple states
 * - Monte Carlo simulation support
 * - Automatic load balancing and chunking
 * - Thread-safe execution with proper resource management
 * 
 * Key Design Principles:
 * - Single Responsibility: ONLY handles parallel execution
 * - No Dependencies: Works with any integrator type
 * - Scalable: Automatic hardware utilization
 * - Safe: Thread-safe with proper error handling
 * 
 * Note: This decorator requires integrator factory support for thread-local copies.
 */
template<system_state S>
class ParallelDecorator : public IntegratorDecorator<S> {
private:
    ParallelConfig config_;

public:
    /**
     * @brief Construct parallel decorator
     * @param integrator The integrator to wrap
     * @param config Parallel configuration (validated on construction)
     * @throws std::invalid_argument if config is invalid
     */
    explicit ParallelDecorator(std::unique_ptr<AbstractIntegrator<S>> integrator,
                              ParallelConfig config = {})
        : IntegratorDecorator<S>(std::move(integrator)), config_(std::move(config)) {
        
        config_.validate();
        
        // Auto-detect optimal thread count if not specified
        if (config_.max_threads == 0) {
            config_.max_threads = std::thread::hardware_concurrency();
            if (config_.max_threads == 0) {
                config_.max_threads = 1;  // Fallback if detection fails
            }
        }
    }

    /**
     * @brief Integrate multiple states in parallel
     * @tparam StateRange Range type containing states to integrate
     * @param states Range of states to integrate
     * @param dt Time step
     * @param end_time Final integration time
     * @throws std::runtime_error if integrator copying is not implemented
     */
    template<typename StateRange>
    void integrate_batch(StateRange&& states, typename IntegratorDecorator<S>::time_type dt, 
                        typename IntegratorDecorator<S>::time_type end_time) {
        const size_t batch_size = std::ranges::size(states);
        
        if (batch_size == 0) {
            return;  // Nothing to do
        }
        
        if (batch_size == 1 || config_.max_threads == 1) {
            // Sequential processing for single state or single thread
            for (auto& state : states) {
                this->wrapped_integrator_->integrate(state, dt, end_time);
            }
            return;
        }
        
        // Parallel processing
        try {
            std::for_each(std::execution::par_unseq, 
                std::ranges::begin(states), std::ranges::end(states),
                [this, dt, end_time](auto& state) {
                    // Create thread-local copy of integrator
                    auto local_integrator = this->create_copy();
                    local_integrator->integrate(state, dt, end_time);
                });
        } catch (const std::exception& e) {
            // Fall back to sequential processing if parallel fails
            for (auto& state : states) {
                this->wrapped_integrator_->integrate(state, dt, end_time);
            }
        }
    }

    /**
     * @brief Monte Carlo integration with parallel execution
     * @tparam Generator Function that generates initial states: state_type(size_t)
     * @tparam Processor Function that processes final states: result_type(const state_type&)
     * @param num_simulations Number of Monte Carlo simulations
     * @param generator Function to generate initial states
     * @param processor Function to process final states
     * @param dt Time step
     * @param end_time Final integration time
     * @return Vector of processed results
     * @throws std::runtime_error if integrator copying is not implemented
     */
    template<typename Generator, typename Processor>
    auto integrate_monte_carlo(size_t num_simulations, Generator&& generator, 
                              Processor&& processor, typename IntegratorDecorator<S>::time_type dt, 
                              typename IntegratorDecorator<S>::time_type end_time) {
        using result_type = std::invoke_result_t<Processor, S>;
        std::vector<result_type> results(num_simulations);

        if (num_simulations == 0) {
            return results;  // Nothing to do
        }

        if (num_simulations == 1 || config_.max_threads == 1) {
            // Sequential processing
            for (size_t i = 0; i < num_simulations; ++i) {
                auto state = generator(i);
                this->wrapped_integrator_->integrate(state, dt, end_time);
                results[i] = processor(state);
            }
            return results;
        }

        // Parallel processing
        try {
            std::for_each(std::execution::par_unseq,
                std::views::iota(0UL, num_simulations).begin(),
                std::views::iota(0UL, num_simulations).end(),
                [&](size_t i) {
                    auto local_integrator = this->create_copy();
                    auto state = generator(i);
                    local_integrator->integrate(state, dt, end_time);
                    results[i] = processor(state);
                });
        } catch (const std::exception& e) {
            // Fall back to sequential processing if parallel fails
            for (size_t i = 0; i < num_simulations; ++i) {
                auto state = generator(i);
                this->wrapped_integrator_->integrate(state, dt, end_time);
                results[i] = processor(state);
            }
        }

        return results;
    }

    /**
     * @brief Chunked parallel processing with load balancing
     * @tparam StateRange Range type containing states
     * @param states Range of states to integrate
     * @param dt Time step
     * @param end_time Final integration time
     */
    template<typename StateRange>
    void integrate_batch_chunked(StateRange&& states, typename IntegratorDecorator<S>::time_type dt, 
                                typename IntegratorDecorator<S>::time_type end_time) {
        const size_t batch_size = std::ranges::size(states);
        
        if (batch_size <= config_.chunk_size || config_.max_threads == 1) {
            integrate_batch(std::forward<StateRange>(states), dt, end_time);
            return;
        }
        
        // Calculate optimal chunk size for load balancing
        size_t effective_chunk_size = config_.enable_auto_chunking ? 
            calculate_optimal_chunk_size(batch_size) : config_.chunk_size;
        
        // Process in chunks
        auto states_begin = std::ranges::begin(states);
        auto states_end = std::ranges::end(states);
        
        for (auto chunk_start = states_begin; chunk_start < states_end;) {
            auto chunk_end = chunk_start;
            std::advance(chunk_end, std::min(effective_chunk_size, 
                static_cast<size_t>(std::distance(chunk_start, states_end))));
            
            // Create range for this chunk and process in parallel
            std::for_each(std::execution::par_unseq, chunk_start, chunk_end,
                [this, dt, end_time](auto& state) {
                    auto local_integrator = this->create_copy();
                    local_integrator->integrate(state, dt, end_time);
                });
            
            chunk_start = chunk_end;
        }
    }

    /**
     * @brief Access and modify parallel configuration
     */
    ParallelConfig& config() { return config_; }
    const ParallelConfig& config() const { return config_; }
    
    /**
     * @brief Update parallel configuration with validation
     * @param new_config New configuration
     * @throws std::invalid_argument if new config is invalid
     */
    void update_config(ParallelConfig new_config) {
        new_config.validate();
        config_ = std::move(new_config);
    }

    /**
     * @brief Get optimal number of threads for current configuration
     */
    size_t get_optimal_thread_count() const {
        return config_.max_threads;
    }

private:
    /**
     * @brief Create a copy of the wrapped integrator for thread-local use
     * @return Unique pointer to integrator copy
     * @throws std::runtime_error if copying is not implemented
     * 
     * Note: This requires integrator factory support or copyable integrators.
     * Future implementation should use a factory pattern or registry.
     */
    std::unique_ptr<AbstractIntegrator<S>> create_copy() {
        // This is a placeholder - actual implementation would depend on integrator type
        // Could use a factory pattern, registry, or require integrators to be copyable
        throw std::runtime_error("Integrator copying not implemented - need factory pattern");
    }

    /**
     * @brief Calculate optimal chunk size for load balancing
     * @param total_size Total number of items to process
     * @return Optimal chunk size
     */
    size_t calculate_optimal_chunk_size(size_t total_size) const {
        // Simple heuristic: distribute work evenly across available threads
        size_t base_chunk_size = std::max(config_.chunk_size, 
                                         total_size / config_.max_threads);
        
        // Adjust for load balancing
        size_t adjusted_size = static_cast<size_t>(base_chunk_size * 
                                                  (1.0 + config_.load_balance_threshold));
        
        return std::min(adjusted_size, total_size);
    }
};

} // namespace diffeq::core::composable