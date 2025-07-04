#pragma once
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/state_creator.hpp>

namespace diffeq::integrators::ode {

/**
 * @brief Improved Euler (Heun's method): y_{n+1} = y_n + h/2 * (k1 + k2)
 * 
 * Second-order explicit method where:
 * - k1 = f(t_n, y_n) 
 * - k2 = f(t_n + h, y_n + h*k1)
 * 
 * Also known as Heun's method or the explicit trapezoidal method.
 * 
 * Order: 2
 * Stability: Better than Euler for most problems
 */
template<system_state S, can_be_time T = double>
class ImprovedEulerIntegrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit ImprovedEulerIntegrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type k1 = StateCreator<state_type>::create(state);
        state_type k2 = StateCreator<state_type>::create(state);
        state_type temp_state = StateCreator<state_type>::create(state);
        
        // k1 = f(t, y)
        this->sys_(this->current_time_, state, k1);
        
        // temp_state = y + dt * k1
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp_state.begin();
            
            temp_it[i] = state_it[i] + dt * k1_it[i];
        }
        
        // k2 = f(t + dt, temp_state)
        this->sys_(this->current_time_ + dt, temp_state, k2);
        
        // y_new = y + dt/2 * (k1 + k2)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            
            state_it[i] = state_it[i] + dt * (k1_it[i] + k2_it[i]) / static_cast<time_type>(2);
        }
        
        this->advance_time(dt);
    }
};

} // namespace diffeq::integrators::ode
