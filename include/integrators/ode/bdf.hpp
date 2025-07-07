#pragma once
#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

namespace diffeq {

/**
 * @brief BDF (Backward Differentiation Formula) integrator
 * 
 * Variable order (1-6) implicit multistep method for stiff systems.
 * Similar to MATLAB's ode15s and scipy's BDF method.
 * Excellent for stiff differential equations.
 * 
 * Order: Variable (1-6)
 * Stages: Variable
 * Adaptive: Yes
 * Stiff: Excellent
 */
template<system_state S>
class BDFIntegrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit BDFIntegrator(system_function sys, 
                          time_type rtol = static_cast<time_type>(1e-6),
                          time_type atol = static_cast<time_type>(1e-9),
                          int max_order = 5)
        : base_type(std::move(sys), rtol, atol), 
          max_order_(std::min<int>(max_order, 6)),
          current_order_(1),
          max_newton_iterations_(10),
          newton_tolerance_(static_cast<time_type>(1e-12)),
          is_initialized_(false) {
        
        // Initialize BDF coefficients
        initialize_bdf_coefficients();
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        if (!is_initialized_) {
            initialize_history(state, dt);
            is_initialized_ = true;
        }
        
        const int max_attempts = 10;
        time_type current_dt = dt;
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            if (bdf_step(state, y_new, error, current_dt)) {
                // Calculate error norm
                time_type err_norm = this->error_norm(error, y_new);
                
                if (err_norm <= 1.0) {
                    // Accept step
                    update_history(y_new, current_dt);
                    adjust_order(err_norm);
                    state = y_new;
                    this->advance_time(current_dt);
                    
                    // Suggest next step size
                    time_type next_dt = this->suggest_step_size(current_dt, err_norm, current_order_ + 1);
                    return std::max<time_type>(this->dt_min_, std::min<time_type>(this->dt_max_, next_dt));
                } else {
                    // Reject step and reduce step size
                    current_dt *= std::max<time_type>(this->safety_factor_ * std::pow(err_norm, -1.0/(current_order_ + 1)), 
                                                     static_cast<time_type>(0.1));
                    current_dt = std::max<time_type>(current_dt, this->dt_min_);
                }
            } else {
                // Newton iteration failed, reduce step size
                current_dt *= static_cast<time_type>(0.5);
                current_dt = std::max<time_type>(current_dt, this->dt_min_);
            }
        }
        
        // If we get here, the step was too difficult - use a simpler approach
        // Fall back to backward Euler (order 1) with a very small step
        return fallback_step(state, dt);
    }

    void set_newton_parameters(int max_iterations, time_type tolerance) {
        max_newton_iterations_ = max_iterations;
        newton_tolerance_ = tolerance;
    }

private:
    int max_order_;
    int current_order_;
    int max_newton_iterations_;
    time_type newton_tolerance_;
    bool is_initialized_;
    
    // History storage
    std::vector<state_type> y_history_;  // Previous solution values
    std::vector<time_type> dt_history_;  // Previous step sizes
    
    // BDF coefficients for different orders
    std::array<std::vector<time_type>, 7> alpha_coeffs_;  // Coefficients for y terms
    std::array<time_type, 7> beta_coeffs_;               // Coefficients for f(y_n+1)
    
    void initialize_bdf_coefficients() {
        // BDF coefficients for orders 1-6
        // alpha[k] contains coefficients for y_{n+1-j}, j=0,1,...,k
        // beta[k] is the coefficient for h*f(t_{n+1}, y_{n+1})
        
        // Order 1 (Backward Euler): y_{n+1} - y_n = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[1] = {1.0, -1.0};
        beta_coeffs_[1] = 1.0;
        
        // Order 2: 3/2*y_{n+1} - 2*y_n + 1/2*y_{n-1} = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[2] = {1.5, -2.0, 0.5};
        beta_coeffs_[2] = 1.0;
        
        // Order 3: 11/6*y_{n+1} - 3*y_n + 3/2*y_{n-1} - 1/3*y_{n-2} = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[3] = {11.0/6.0, -3.0, 1.5, -1.0/3.0};
        beta_coeffs_[3] = 1.0;
        
        // Order 4: 25/12*y_{n+1} - 4*y_n + 3*y_{n-1} - 4/3*y_{n-2} + 1/4*y_{n-3} = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[4] = {25.0/12.0, -4.0, 3.0, -4.0/3.0, 0.25};
        beta_coeffs_[4] = 1.0;
        
        // Order 5: 137/60*y_{n+1} - 5*y_n + 5*y_{n-1} - 10/3*y_{n-2} + 5/4*y_{n-3} - 1/5*y_{n-4} = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[5] = {137.0/60.0, -5.0, 5.0, -10.0/3.0, 1.25, -0.2};
        beta_coeffs_[5] = 1.0;
        
        // Order 6: 147/60*y_{n+1} - 6*y_n + 15/2*y_{n-1} - 20/3*y_{n-2} + 15/4*y_{n-3} - 6/5*y_{n-4} + 1/6*y_{n-5} = h*f(t_{n+1}, y_{n+1})
        alpha_coeffs_[6] = {147.0/60.0, -6.0, 7.5, -20.0/3.0, 3.75, -1.2, 1.0/6.0};
        beta_coeffs_[6] = 1.0;
    }
    
    void initialize_history(const state_type& y0, time_type dt) {
        y_history_.clear();
        dt_history_.clear();
        y_history_.push_back(y0);
        dt_history_.push_back(dt);
        current_order_ = 1;
    }
    
    void update_history(const state_type& y_new, time_type dt) {
        y_history_.insert(y_history_.begin(), y_new);
        dt_history_.insert(dt_history_.begin(), dt);
        
        // Keep only the required number of history points
        while (y_history_.size() > static_cast<size_t>(max_order_ + 1)) {
            y_history_.pop_back();
            dt_history_.pop_back();
        }
    }
    
    void adjust_order(time_type error_norm) {
        // Simple order adjustment strategy
        if (error_norm < 0.1 && current_order_ < max_order_ && 
            y_history_.size() >= static_cast<size_t>(current_order_ + 1)) {
            current_order_++;
        } else if (error_norm > 0.5 && current_order_ > 1) {
            current_order_--;
        }
    }
    
    bool bdf_step(const state_type& y_current, state_type& y_new, state_type& error, time_type dt) {
        // Initial guess for Newton iteration (use extrapolation)
        y_new = y_history_[0];  // Use most recent value as initial guess
        
        // Newton iteration to solve the implicit BDF equation
        for (int iter = 0; iter < max_newton_iterations_; ++iter) {
            state_type f_new = StateCreator<state_type>::create(y_new);
            state_type residual = StateCreator<state_type>::create(y_new);
            
            // Evaluate f(t_{n+1}, y_{n+1})
            this->sys_(this->current_time_ + dt, y_new, f_new);
            
            // Calculate residual: R = alpha[0]*y_{n+1} + sum(alpha[j]*y_{n+1-j}) - beta*h*f(t_{n+1}, y_{n+1})
            for (std::size_t i = 0; i < y_new.size(); ++i) {
                residual[i] = alpha_coeffs_[current_order_][0] * y_new[i];
                
                // Add history terms
                for (int j = 1; j <= current_order_ && j < static_cast<int>(y_history_.size()); ++j) {
                    residual[i] += alpha_coeffs_[current_order_][j] * y_history_[j][i];
                }
                
                // Subtract the f term
                residual[i] -= beta_coeffs_[current_order_] * dt * f_new[i];
            }
            
            // Check convergence
            time_type residual_norm = 0;
            for (std::size_t i = 0; i < residual.size(); ++i) {
                residual_norm += residual[i] * residual[i];
            }
            residual_norm = std::sqrt(residual_norm);
            
            if (residual_norm < newton_tolerance_) {
                // Newton iteration converged
                calculate_error_estimate(y_new, error, dt);
                return true;
            }
            
            // Update y_new using simplified Newton step
            // For simplicity, we use a diagonal approximation of the Jacobian
            for (std::size_t i = 0; i < y_new.size(); ++i) {
                // Simplified Newton update: y_new -= residual / (alpha[0] - beta*h*df/dy)
                // We approximate df/dy using finite differences
                time_type df_dy = estimate_jacobian_diagonal(i, y_new, dt);
                time_type denominator = alpha_coeffs_[current_order_][0] - beta_coeffs_[current_order_] * dt * df_dy;
                
                if (std::abs(denominator) > newton_tolerance_) {
                    y_new[i] -= residual[i] / denominator;
                }
            }
        }
        
        // Newton iteration failed to converge
        return false;
    }
    
    time_type estimate_jacobian_diagonal(std::size_t i, const state_type& y, time_type dt) {
        // Estimate diagonal element of Jacobian using finite differences
        time_type epsilon = static_cast<time_type>(1e-8);
        state_type y_pert = StateCreator<state_type>::create(y);
        state_type f_orig = StateCreator<state_type>::create(y);
        state_type f_pert = StateCreator<state_type>::create(y);
        
        // Evaluate f at original point
        this->sys_(this->current_time_ + dt, y, f_orig);
        
        // Perturb y[i] and evaluate f
        y_pert = y;
        y_pert[i] += epsilon;
        this->sys_(this->current_time_ + dt, y_pert, f_pert);
        
        // Estimate ∂f_i/∂y_i
        return (f_pert[i] - f_orig[i]) / epsilon;
    }
    
    void calculate_error_estimate(const state_type& y_new, state_type& error, time_type dt) {
        // Simple error estimate based on the difference between current and previous order solutions
        if (y_history_.size() >= 2 && current_order_ > 1) {
            // Use the difference between current order and previous order as error estimate
            state_type y_prev_order = StateCreator<state_type>::create(y_new);
            
            // Reconstruct solution using previous order
            for (std::size_t i = 0; i < y_new.size(); ++i) {
                y_prev_order[i] = alpha_coeffs_[current_order_ - 1][0] * y_new[i];
                
                // Add history terms for previous order
                for (int j = 1; j < current_order_ && j < static_cast<int>(y_history_.size()); ++j) {
                    y_prev_order[i] += alpha_coeffs_[current_order_ - 1][j] * y_history_[j][i];
                }
                
                // Calculate error as difference
                error[i] = std::abs(y_new[i] - y_prev_order[i]);
            }
        } else {
            // Fallback error estimate
            for (std::size_t i = 0; i < y_new.size(); ++i) {
                error[i] = static_cast<time_type>(1e-6);
            }
        }
    }
    
    time_type fallback_step(state_type& state, time_type dt) {
        // Fallback to backward Euler with very small step
        time_type small_dt = std::min<time_type>(dt, static_cast<time_type>(1e-6));
        
        state_type y_new = StateCreator<state_type>::create(state);
        state_type f_new = StateCreator<state_type>::create(state);
        
        // Simple backward Euler iteration
        y_new = state;
        for (int iter = 0; iter < 5; ++iter) {
            this->sys_(this->current_time_ + small_dt, y_new, f_new);
            
            for (std::size_t i = 0; i < state.size(); ++i) {
                y_new[i] = state[i] + small_dt * f_new[i];
            }
        }
        
        state = y_new;
        this->advance_time(small_dt);
        
        return small_dt;
    }
};

} // namespace diffeq
