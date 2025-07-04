#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq::sde {

/**
 * @brief Stochastic Runge-Kutta method (SRI1)
 * 
 * A second-order strong Taylor scheme for SDEs.
 * More stable than Milstein for certain problems.
 * 
 * Strong order: 1.0
 * Weak order: 2.0
 */
template<system_state StateType, can_be_time TimeType>
class SRI1Integrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    explicit SRI1Integrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                          std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener) {}
    
    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type k1 = StateCreator<state_type>::create(state);
        state_type k2 = StateCreator<state_type>::create(state);
        state_type g1 = StateCreator<state_type>::create(state);
        state_type g2 = StateCreator<state_type>::create(state);
        state_type temp_state = StateCreator<state_type>::create(state);
        state_type dW = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        time_type t = this->current_time_;
        value_type sqrt_dt = std::sqrt(static_cast<value_type>(dt));
        
        // Stage 1: k1 = f(t, X), g1 = g(t, X)
        this->problem_->drift(t, state, k1);
        this->problem_->diffusion(t, state, g1);
        
        // Intermediate state for stage 2
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto g1_it = g1.begin();
            auto temp_it = temp_state.begin();
            auto dW_it = dW.begin();
            
            temp_it[i] = state_it[i] + k1_it[i] * dt + g1_it[i] * sqrt_dt;
        }
        
        // Stage 2: k2 = f(t + dt, temp_state), g2 = g(t + dt, temp_state)
        this->problem_->drift(t + dt, temp_state, k2);
        this->problem_->diffusion(t + dt, temp_state, g2);
        
        // Apply noise to diffusion terms
        state_type g1_noise = StateCreator<state_type>::create(state);
        state_type g2_noise = StateCreator<state_type>::create(state);
        
        for (size_t i = 0; i < state.size(); ++i) {
            auto g1_it = g1.begin();
            auto g2_it = g2.begin();
            auto g1_noise_it = g1_noise.begin();
            auto g2_noise_it = g2_noise.begin();
            
            g1_noise_it[i] = g1_it[i];
            g2_noise_it[i] = g2_it[i];
        }
        
        this->problem_->apply_noise(t, state, g1_noise, dW);
        this->problem_->apply_noise(t + dt, temp_state, g2_noise, dW);
        
        // Final update: X_{n+1} = X_n + (k1 + k2)/2 * dt + (g1 + g2)/2 * dW
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto g1_noise_it = g1_noise.begin();
            auto g2_noise_it = g2_noise.begin();
            
            state_it[i] += (k1_it[i] + k2_it[i]) * dt * static_cast<value_type>(0.5) +
                          (g1_noise_it[i] + g2_noise_it[i]) * static_cast<value_type>(0.5);
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "SRI1 (Stochastic Runge-Kutta)";
    }
};

} // namespace diffeq::sde
