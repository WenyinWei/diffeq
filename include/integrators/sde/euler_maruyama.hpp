#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq {

/**
 * @brief Euler-Maruyama method for SDEs
 * 
 * The most basic SDE solver. Explicit scheme:
 * X_{n+1} = X_n + f(t_n, X_n) * dt + g(t_n, X_n) * dW_n
 * 
 * Strong order: 0.5
 * Weak order: 1.0
 */
template<system_state StateType>
class EulerMaruyamaIntegrator : public AbstractSDEIntegrator<StateType> {
public:
    using base_type = AbstractSDEIntegrator<StateType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    explicit EulerMaruyamaIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                                   std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener) {}
    
    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type drift_term = StateCreator<state_type>::create(state);
        state_type diffusion_term = StateCreator<state_type>::create(state);
        state_type dW = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        // Compute drift: f(t, X)
        this->problem_->drift(this->current_time_, state, drift_term);
        
        // Compute diffusion: g(t, X)
        this->problem_->diffusion(this->current_time_, state, diffusion_term);
        
        // Apply noise to diffusion term
        this->problem_->apply_noise(this->current_time_, state, diffusion_term, dW);
        
        // Update state: X_{n+1} = X_n + f*dt + g*dW
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto drift_it = drift_term.begin();
            auto diffusion_it = diffusion_term.begin();
            
            state_it[i] += drift_it[i] * dt + diffusion_it[i];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "Euler-Maruyama";
    }
};

} // namespace diffeq
