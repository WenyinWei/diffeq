#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <algorithm>
#include <cmath>

namespace diffeq::sde {

/**
 * @brief Implicit Euler-Maruyama method
 * 
 * Implicit version for better stability with stiff SDEs:
 * X_{n+1} = X_n + f(t_{n+1}, X_{n+1}) * dt + g(t_n, X_n) * dW_n
 * 
 * Uses fixed-point iteration to solve the implicit equation.
 */
template<system_state StateType, can_be_time TimeType>
class ImplicitEulerMaruyamaIntegrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    explicit ImplicitEulerMaruyamaIntegrator(
        std::shared_ptr<typename base_type::sde_problem_type> problem,
        std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr,
        int max_iterations = 10,
        value_type tolerance = 1e-8)
        : base_type(problem, wiener)
        , max_iterations_(max_iterations)
        , tolerance_(tolerance) {}
    
    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type diffusion_term = StateCreator<state_type>::create(state);
        state_type dW = StateCreator<state_type>::create(state);
        state_type x_new = StateCreator<state_type>::create(state);
        state_type x_old = StateCreator<state_type>::create(state);
        state_type drift_term = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        // Compute explicit diffusion term: g(t_n, X_n) * dW_n
        this->problem_->diffusion(this->current_time_, state, diffusion_term);
        this->problem_->apply_noise(this->current_time_, state, diffusion_term, dW);
        
        // Initial guess: x_new = x_old (explicit Euler)
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto x_new_it = x_new.begin();
            auto diffusion_it = diffusion_term.begin();
            
            x_new_it[i] = state_it[i] + diffusion_it[i];
        }
        
        // Fixed-point iteration to solve: x_new = x_old + f(t+dt, x_new)*dt + diffusion_term
        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Save old iterate
            for (size_t i = 0; i < state.size(); ++i) {
                auto x_new_it = x_new.begin();
                auto x_old_it = x_old.begin();
                x_old_it[i] = x_new_it[i];
            }
            
            // Compute drift at new time and new state
            this->problem_->drift(this->current_time_ + dt, x_old, drift_term);
            
            // Update: x_new = x_n + f(t+dt, x_old)*dt + diffusion_term
            value_type max_change = 0;
            for (size_t i = 0; i < state.size(); ++i) {
                auto state_it = state.begin();
                auto x_new_it = x_new.begin();
                auto x_old_it = x_old.begin();
                auto drift_it = drift_term.begin();
                auto diffusion_it = diffusion_term.begin();
                
                value_type new_val = state_it[i] + drift_it[i] * dt + diffusion_it[i];
                value_type change = std::abs(new_val - x_old_it[i]);
                max_change = std::max(max_change, change);
                x_new_it[i] = new_val;
            }
            
            // Check convergence
            if (max_change < tolerance_) {
                break;
            }
        }
        
        // Update state
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto x_new_it = x_new.begin();
            state_it[i] = x_new_it[i];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "Implicit Euler-Maruyama";
    }
    
    void set_iteration_parameters(int max_iterations, value_type tolerance) {
        max_iterations_ = max_iterations;
        tolerance_ = tolerance;
    }

private:
    int max_iterations_;
    value_type tolerance_;
};

} // namespace diffeq::sde
