#pragma once

#include "integrator_decorator.hpp"
#include "timeout_decorator.hpp"
#include "parallel_decorator.hpp"
#include "async_decorator.hpp"
#include "output_decorator.hpp"
#include "signal_decorator.hpp"
#include "interpolation_decorator.hpp"
#include "interprocess_decorator.hpp"
#include "event_decorator.hpp"
#include <memory>

namespace diffeq::core::composable {

/**
 * @brief Builder for composing multiple facilities
 * 
 * This allows flexible combination of any facilities without
 * exponential class combinations. Uses the decorator pattern
 * to stack facilities in any order.
 * 
 * Key Design Principles:
 * - Fluent Interface: Chainable method calls
 * - Order Independence: Facilities work in any composition order  
 * - Type Safety: Compile-time type checking
 * - Extensibility: Easy to add new facilities without modification
 * 
 * Example Usage:
 * ```cpp
 * auto integrator = make_builder(base_integrator)
 *     .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::seconds{30}})
 *     .with_parallel(ParallelConfig{.max_threads = 8})
 *     .with_async()
 *     .with_signals()
 *     .with_output(OutputConfig{.mode = OutputMode::HYBRID})
 *     .build();
 * ```
 */
template<system_state S, can_be_time T = double>
class IntegratorBuilder {
private:
    std::unique_ptr<AbstractIntegrator<S, T>> integrator_;

public:
    /**
     * @brief Construct builder with base integrator
     * @param integrator The integrator to build upon (takes ownership)
     */
    explicit IntegratorBuilder(std::unique_ptr<AbstractIntegrator<S, T>> integrator)
        : integrator_(std::move(integrator)) {
        
        if (!integrator_) {
            throw std::invalid_argument("Base integrator cannot be null");
        }
    }

    /**
     * @brief Add timeout protection facility
     * @param config Timeout configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_timeout(TimeoutConfig config = {}) {
        integrator_ = std::make_unique<TimeoutDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add parallel execution facility
     * @param config Parallel configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_parallel(ParallelConfig config = {}) {
        integrator_ = std::make_unique<ParallelDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add async execution facility
     * @param config Async configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_async(AsyncConfig config = {}) {
        integrator_ = std::make_unique<AsyncDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add output handling facility
     * @param config Output configuration (uses defaults if not specified)
     * @param handler Optional output handler function
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_output(OutputConfig config = {}, 
                                  std::function<void(const S&, T, size_t)> handler = nullptr) {
        integrator_ = std::make_unique<OutputDecorator<S, T>>(
            std::move(integrator_), std::move(config), std::move(handler));
        return *this;
    }

    /**
     * @brief Add signal processing facility
     * @param config Signal configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_signals(SignalConfig config = {}) {
        integrator_ = std::make_unique<SignalDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add interpolation/dense output facility
     * @param config Interpolation configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_interpolation(InterpolationConfig config = {}) {
        integrator_ = std::make_unique<InterpolationDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add interprocess communication facility
     * @param config Interprocess configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_interprocess(InterprocessConfig config = {}) {
        integrator_ = std::make_unique<InterprocessDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Add event-driven feedback facility
     * @param config Event configuration (uses defaults if not specified)
     * @return Reference to this builder for chaining
     * @throws std::invalid_argument if config is invalid
     */
    IntegratorBuilder& with_events(EventConfig config = {}) {
        integrator_ = std::make_unique<EventDecorator<S, T>>(
            std::move(integrator_), std::move(config));
        return *this;
    }

    /**
     * @brief Build the final composed integrator
     * @return Unique pointer to the composed integrator
     * 
     * Note: After calling build(), the builder is left in a valid but unspecified state.
     * Do not use the builder after calling build().
     */
    std::unique_ptr<AbstractIntegrator<S, T>> build() {
        if (!integrator_) {
            throw std::runtime_error("Builder has already been used or is in invalid state");
        }
        return std::move(integrator_);
    }

    /**
     * @brief Get specific decorator type from the composition chain
     * @tparam DecoratorType The specific decorator type to retrieve
     * @return Pointer to the decorator, or nullptr if not found
     * 
     * Note: This performs a dynamic_cast and may be expensive. Use sparingly.
     * 
     * Example:
     * ```cpp
     * auto builder = make_builder(base).with_timeout().with_async();
     * auto* timeout_decorator = builder.get_as<TimeoutDecorator<S, T>>();
     * if (timeout_decorator) {
     *     // Access timeout-specific functionality
     *     timeout_decorator->config().timeout_duration = std::chrono::seconds{60};
     * }
     * ```
     */
    template<typename DecoratorType>
    DecoratorType* get_as() {
        return dynamic_cast<DecoratorType*>(integrator_.get());
    }

    /**
     * @brief Check if the builder has a valid integrator
     * @return true if the builder can still be used
     */
    bool is_valid() const {
        return integrator_ != nullptr;
    }

    /**
     * @brief Get information about the current composition
     * @return String describing the current decorator stack
     */
    std::string get_composition_info() const {
        if (!integrator_) {
            return "Builder is empty or has been built";
        }
        
        std::string info = "Composition: ";
        
        // Try to identify decorators in the chain
        // This is a simplified version - a real implementation might maintain
        // a list of applied decorators for better introspection
        
        if (dynamic_cast<TimeoutDecorator<S, T>*>(integrator_.get())) {
            info += "Timeout -> ";
        }
        if (dynamic_cast<ParallelDecorator<S, T>*>(integrator_.get())) {
            info += "Parallel -> ";
        }
        if (dynamic_cast<AsyncDecorator<S, T>*>(integrator_.get())) {
            info += "Async -> ";
        }
        if (dynamic_cast<OutputDecorator<S, T>*>(integrator_.get())) {
            info += "Output -> ";
        }
        if (dynamic_cast<SignalDecorator<S, T>*>(integrator_.get())) {
            info += "Signal -> ";
        }
        if (dynamic_cast<InterpolationDecorator<S, T>*>(integrator_.get())) {
            info += "Interpolation -> ";
        }
        if (dynamic_cast<InterprocessDecorator<S, T>*>(integrator_.get())) {
            info += "Interprocess -> ";
        }
        if (dynamic_cast<EventDecorator<S, T>*>(integrator_.get())) {
            info += "Events -> ";
        }
        
        info += "Base";
        return info;
    }
};

// ============================================================================
// FACTORY FUNCTIONS (Easy creation)
// ============================================================================

/**
 * @brief Create a builder starting with any integrator
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator to build upon
 * @return IntegratorBuilder for fluent composition
 * @throws std::invalid_argument if integrator is null
 */
template<system_state S, can_be_time T = double>
auto make_builder(std::unique_ptr<AbstractIntegrator<S, T>> integrator) {
    return IntegratorBuilder<S, T>(std::move(integrator));
}

/**
 * @brief Create a builder starting with a copied integrator
 * @tparam Integrator Integrator type (must be copyable)
 * @param integrator Integrator to copy and build upon
 * @return IntegratorBuilder for fluent composition
 */
template<typename Integrator>
auto make_builder_copy(const Integrator& integrator) {
    return IntegratorBuilder<typename Integrator::state_type, typename Integrator::time_type>(
        std::make_unique<Integrator>(integrator));
}

// ============================================================================
// CONVENIENCE FUNCTIONS (Common single-decorator use cases)
// ============================================================================

/**
 * @brief Create integrator with only timeout protection
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Timeout configuration
 * @return Timeout-protected integrator
 */
template<system_state S, can_be_time T = double>
auto with_timeout_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator, 
                      TimeoutConfig config = {}) {
    return make_builder(std::move(integrator)).with_timeout(std::move(config)).build();
}

/**
 * @brief Create integrator with only parallel execution
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Parallel configuration
 * @return Parallel-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_parallel_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                       ParallelConfig config = {}) {
    return make_builder(std::move(integrator)).with_parallel(std::move(config)).build();
}

/**
 * @brief Create integrator with only async execution
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Async configuration
 * @return Async-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_async_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                     AsyncConfig config = {}) {
    return make_builder(std::move(integrator)).with_async(std::move(config)).build();
}

/**
 * @brief Create integrator with only interpolation/dense output
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Interpolation configuration
 * @return Interpolation-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_interpolation_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                             InterpolationConfig config = {}) {
    return make_builder(std::move(integrator)).with_interpolation(std::move(config)).build();
}

/**
 * @brief Create integrator with only interprocess communication
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Interprocess configuration
 * @return Interprocess-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_interprocess_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                            InterprocessConfig config = {}) {
    return make_builder(std::move(integrator)).with_interprocess(std::move(config)).build();
}

/**
 * @brief Create integrator with only event-driven feedback
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Event configuration
 * @return Event-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_events_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                      EventConfig config = {}) {
    return make_builder(std::move(integrator)).with_events(std::move(config)).build();
}

/**
 * @brief Create integrator with only output handling
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Output configuration
 * @param handler Optional output handler
 * @return Output-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_output_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                     OutputConfig config = {},
                     std::function<void(const S&, T, size_t)> handler = nullptr) {
    return make_builder(std::move(integrator))
        .with_output(std::move(config), std::move(handler)).build();
}

/**
 * @brief Create integrator with only signal processing
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param config Signal configuration
 * @return Signal-enabled integrator
 */
template<system_state S, can_be_time T = double>
auto with_signals_only(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                      SignalConfig config = {}) {
    return make_builder(std::move(integrator)).with_signals(std::move(config)).build();
}

// ============================================================================
// COMMON COMPOSITIONS (Frequently used combinations)
// ============================================================================

/**
 * @brief Create integrator for real-time applications
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param timeout_ms Timeout in milliseconds
 * @return Integrator with timeout + async + signals
 */
template<system_state S, can_be_time T = double>
auto for_realtime(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                  std::chrono::milliseconds timeout_ms = std::chrono::milliseconds{100}) {
    return make_builder(std::move(integrator))
        .with_timeout(TimeoutConfig{.timeout_duration = timeout_ms})
        .with_async()
        .with_signals()
        .build();
}

/**
 * @brief Create integrator for research/batch processing
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @param max_threads Maximum parallel threads
 * @return Integrator with timeout + parallel + output
 */
template<system_state S, can_be_time T = double>
auto for_research(std::unique_ptr<AbstractIntegrator<S, T>> integrator,
                  size_t max_threads = 0) {
    return make_builder(std::move(integrator))
        .with_timeout(TimeoutConfig{.timeout_duration = std::chrono::hours{24}})
        .with_parallel(ParallelConfig{.max_threads = max_threads})
        .with_output(OutputConfig{.mode = OutputMode::OFFLINE})
        .build();
}

/**
 * @brief Create integrator for production servers
 * @tparam S State type
 * @tparam T Time type
 * @param integrator Base integrator
 * @return Integrator with safe timeout + monitoring output
 */
template<system_state S, can_be_time T = double>
auto for_production(std::unique_ptr<AbstractIntegrator<S, T>> integrator) {
    return make_builder(std::move(integrator))
        .with_timeout(TimeoutConfig{
            .timeout_duration = std::chrono::seconds{30},
            .throw_on_timeout = false  // Don't crash server
        })
        .with_output(OutputConfig{.mode = OutputMode::HYBRID})
        .build();
}

} // namespace diffeq::core::composable