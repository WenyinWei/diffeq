#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq::sde {

/**
 * @brief Milstein method for SDEs
 * 
 * First-order strong method with correction term:
 * X_{n+1} = X_n + f(t_n, X_n) * dt + g(t_n, X_n) * dW_n + 
 *           0.5 * g(t_n, X_n) * g'(t_n, X_n) * (dW_n^2 - dt)
 * 
 * Strong order: 1.0
 * Weak order: 1.0
 * 
 * Note: Requires derivative of diffusion function g'(t, X)
 */
template<system_state StateType>
class MilsteinIntegrator : public AbstractSDEIntegrator<StateType> {
public:
    using base_type = AbstractSDEIntegrator<StateType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    // Function signature for diffusion derivative
    using diffusion_derivative_function = std::function<void(time_type, const state_type&, state_type&)>;
    
    explicit MilsteinIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                               diffusion_derivative_function diffusion_derivative,
                               std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener)
        , diffusion_derivative_(std::move(diffusion_derivative)) {}
    
    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type drift_term = StateCreator<state_type>::create(state);
        state_type diffusion_term = StateCreator<state_type>::create(state);
        state_type diffusion_deriv_term = StateCreator<state_type>::create(state);
        state_type dW = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        // Compute drift: f(t, X)
        this->problem_->drift(this->current_time_, state, drift_term);
        
        // Compute diffusion: g(t, X)
        this->problem_->diffusion(this->current_time_, state, diffusion_term);
        
        // Compute diffusion derivative: g'(t, X)
        diffusion_derivative_(this->current_time_, state, diffusion_deriv_term);
        
        // Apply noise to diffusion term
        this->problem_->apply_noise(this->current_time_, state, diffusion_term, dW);
        
        // Update state: X_{n+1} = X_n + f*dt + g*dW + 0.5*g*g'*(dW^2 - dt)
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto drift_it = drift_term.begin();
            auto diffusion_it = diffusion_term.begin();
            auto diffusion_deriv_it = diffusion_deriv_term.begin();
            auto dW_it = dW.begin();
            
            value_type dW_squared = dW_it[i] * dW_it[i];
            value_type correction = static_cast<value_type>(0.5) * diffusion_it[i] * diffusion_deriv_it[i] * (dW_squared - dt);
            
            state_it[i] += drift_it[i] * dt + diffusion_it[i] + correction;
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "Milstein";
    }

private:
    diffusion_derivative_function diffusion_derivative_;
};

} // namespace diffeq::sde
