#pragma once
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/state_creator.hpp>

namespace diffeq::integrators::ode {

/**
 * @brief Simple Euler integrator: y_{n+1} = y_n + h * f(t_n, y_n)
 * 
 * First-order explicit method for ODEs.
 * Simple but not very accurate - mainly for educational purposes.
 * 
 * Order: 1
 * Stability: Conditionally stable
 */
template<system_state S, can_be_time T = double>
class EulerIntegrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit EulerIntegrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary state for derivative
        state_type dydt = StateCreator<state_type>::create(state);
        
        // Compute derivative: dydt = f(t, y)
        this->sys_(this->current_time_, state, dydt);
        
        // Update state: y_new = y + dt * dydt
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto dydt_it = dydt.begin();
            
            state_it[i] = state_it[i] + dt * dydt_it[i];
        }
        
        this->advance_time(dt);
    }
};

} // namespace diffeq::integrators::ode
