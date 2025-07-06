#pragma once

#include "../concepts.hpp"
#include "../abstract_integrator.hpp"
#include <memory>

namespace diffeq::core::composable {

/**
 * @brief Base decorator interface for integrator enhancements
 * 
 * This provides the foundation for the decorator pattern, allowing
 * facilities to be stacked independently without tight coupling.
 * 
 * Design Principles:
 * - High Cohesion: Each decorator focuses on single responsibility
 * - Low Coupling: Decorators combine without dependencies
 * - Delegation: Forwards calls to wrapped integrator by default
 * - Extensibility: Easy to add new decorators without modification
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
    /**
     * @brief Construct decorator wrapping another integrator
     * @param integrator The integrator to wrap (takes ownership)
     */
    explicit IntegratorDecorator(std::unique_ptr<base_type> integrator)
        : base_type(integrator->sys_), wrapped_integrator_(std::move(integrator)) {}

    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~IntegratorDecorator() = default;

    // Delegate core functionality by default - decorators override as needed
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

    /**
     * @brief Access to wrapped integrator for advanced use
     * @return Reference to the wrapped integrator
     */
    base_type& wrapped() { return *wrapped_integrator_; }
    const base_type& wrapped() const { return *wrapped_integrator_; }

    /**
     * @brief Check if wrapped integrator exists
     * @return true if wrapped integrator is valid
     */
    bool has_wrapped_integrator() const { return wrapped_integrator_ != nullptr; }
};

} // namespace diffeq::core::composable