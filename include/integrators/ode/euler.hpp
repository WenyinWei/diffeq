#pragma once
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/state_creator.hpp>

namespace diffeq {

/**
 * @brief Forward Euler integrator
 * 
 * First-order explicit method. Simple but often unstable.
 * 
 * Order: 1
 * Stability: Poor for stiff problems
 * Usage: Educational purposes, simple problems
 */
template<system_state S>
class EulerIntegrator : public core::AbstractIntegrator<S> {
public:
    using base_type = core::AbstractIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit EulerIntegrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary state for derivative
        state_type derivative = StateCreator<state_type>::create(state);
        
        // Compute derivative: f(t, y)
        this->sys_(this->current_time_, state, derivative);
        
        // Update state: y_{n+1} = y_n + dt * f(t_n, y_n)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto deriv_it = derivative.begin();
            state_it[i] += dt * deriv_it[i];
        }
        
        this->advance_time(dt);
    }
};

} // namespace diffeq
