#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq::sde {

/**
 * @brief Milstein method for SDEs
 * 
 * Higher-order method that includes the Lévy area term:
 * X_{n+1} = X_n + f*dt + g*dW + (1/2)*g*g'*(dW)^2 - (1/2)*g*g'*dt
 * 
 * Strong order: 1.0
 * Weak order: 1.0
 * 
 * Note: This implementation assumes diagonal noise for simplicity.
 * For general noise, the Lévy area computation is more complex.
 */
template<system_state StateType, can_be_time TimeType>
class MilsteinIntegrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    explicit MilsteinIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                              std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr,
                              value_type finite_diff_eps = 1e-6)
        : base_type(problem, wiener)
        , finite_diff_eps_(finite_diff_eps) {}
    
    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type drift_term = StateCreator<state_type>::create(state);
        state_type diffusion_term = StateCreator<state_type>::create(state);
        state_type levy_term = StateCreator<state_type>::create(state);
        state_type dW = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        // Compute drift: f(t, X)
        this->problem_->drift(this->current_time_, state, drift_term);
        
        // Compute diffusion: g(t, X)
        this->problem_->diffusion(this->current_time_, state, diffusion_term);
        
        // Compute Lévy area term: (1/2) * g * (∂g/∂x) * ((dW)^2 - dt)
        compute_levy_area_term(state, diffusion_term, dW, dt, levy_term);
        
        // Apply noise to diffusion term
        this->problem_->apply_noise(this->current_time_, state, diffusion_term, dW);
        
        // Update state: X_{n+1} = X_n + f*dt + g*dW + levy_term
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto drift_it = drift_term.begin();
            auto diffusion_it = diffusion_term.begin();
            auto levy_it = levy_term.begin();
            
            state_it[i] += drift_it[i] * dt + diffusion_it[i] + levy_it[i];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "Milstein";
    }

private:
    value_type finite_diff_eps_;
    
    void compute_levy_area_term(const state_type& state, const state_type& g,
                               const state_type& dW, time_type dt, state_type& levy_term) {
        // For diagonal noise: levy = (1/2) * g * (∂g/∂x) * ((dW)^2 - dt)
        // We approximate ∂g/∂x using finite differences
        
        state_type g_plus = StateCreator<state_type>::create(state);
        state_type state_plus = StateCreator<state_type>::create(state);
        
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto state_plus_it = state_plus.begin();
            auto g_it = g.begin();
            auto g_plus_it = g_plus.begin();
            auto dW_it = dW.begin();
            auto levy_it = levy_term.begin();
            
            // Create perturbed state
            for (size_t j = 0; j < state.size(); ++j) {
                state_plus_it[j] = state_it[j];
            }
            state_plus_it[i] += finite_diff_eps_;
            
            // Compute g at perturbed state
            this->problem_->diffusion(this->current_time_, state_plus, g_plus);
            
            // Finite difference approximation: ∂g_i/∂x_i ≈ (g_i(x+ε) - g_i(x))/ε
            value_type dg_dx = (g_plus_it[i] - g_it[i]) / finite_diff_eps_;
            
            // Lévy area term: (1/2) * g_i * (∂g_i/∂x_i) * ((dW_i)^2 - dt)
            value_type dW_squared_minus_dt = dW_it[i] * dW_it[i] - static_cast<value_type>(dt);
            levy_it[i] = static_cast<value_type>(0.5) * g_it[i] * dg_dx * dW_squared_minus_dt;
        }
    }
};

} // namespace diffeq::sde
