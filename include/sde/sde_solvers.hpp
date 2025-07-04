#pragma once

#include <sde/sde_base.hpp>
#include <sde/advanced_sde_solvers.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq::sde {

/**
 * @brief Euler-Maruyama method for SDEs
 * 
 * The most basic SDE solver. Explicit scheme:
 * X_{n+1} = X_n + f(t_n, X_n) * dt + g(t_n, X_n) * dW_n
 * 
 * Strong order: 0.5
 * Weak order: 1.0
 */
template<system_state StateType, can_be_time TimeType>
class EulerMaruyamaIntegrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
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
