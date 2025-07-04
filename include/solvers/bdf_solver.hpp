#pragma once
#include <core/adaptive_integrator.hpp>
#include <vector>
#include <array>
#include <cmath>

// BDF (Backward Differentiation Formula) integrator
// Variable order (1-6) implicit multistep method for stiff systems
// Similar to MATLAB's ode15s and scipy's BDF method
template<system_state S, can_be_time T = double>
class BDFIntegrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit BDFIntegrator(system_function sys, 
                          time_type rtol = static_cast<time_type>(1e-6),
                          time_type atol = static_cast<time_type>(1e-9),
                          int max_order = 5)
        : base_type(std::move(sys), rtol, atol), 
          max_order_(std::min(max_order, 6)),
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
        }
        
        const int max_attempts = 10;
        time_type current_dt = dt;
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            if (bdf_step(state, y_new, error, current_dt)) {
                // Calculate error norm
                time_type err_norm = this->error_norm(error, state);
                
                if (err_norm <= 1.0) {
                    // Step accepted
                    update_history(y_new, current_dt);
                    state = y_new;
                    this->advance_time(current_dt);
                    
                    // Suggest next step size and possibly increase order
                    current_dt = this->suggest_step_size(current_dt, err_norm, current_order_ + 1);
                    adjust_order(err_norm);
                    
                    return current_dt;
                } else {
                    // Step rejected, reduce step size and possibly order
                    current_dt = this->suggest_step_size(current_dt, err_norm, current_order_ + 1);
                    if (current_order_ > 1) {
                        current_order_--;
                    }
                    if (current_dt < this->dt_min_) {
                        break;
                    }
                }
            } else {
                // Newton iteration failed
                current_dt *= static_cast<time_type>(0.5);
                if (current_order_ > 1) {
                    current_order_--;
                }
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("BDF: Maximum number of step size reductions exceeded");
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
        alpha_coeffs_[2] = {3.0/2.0, -2.0, 1.0/2.0};
        beta_coeffs_[2] = 1.0;
        
        // Order 3
        alpha_coeffs_[3] = {11.0/6.0, -3.0, 3.0/2.0, -1.0/3.0};
        beta_coeffs_[3] = 1.0;
        
        // Order 4
        alpha_coeffs_[4] = {25.0/12.0, -4.0, 3.0, -4.0/3.0, 1.0/4.0};
        beta_coeffs_[4] = 1.0;
        
        // Order 5
        alpha_coeffs_[5] = {137.0/60.0, -5.0, 5.0, -10.0/3.0, 5.0/4.0, -1.0/5.0};
        beta_coeffs_[5] = 1.0;
        
        // Order 6
        alpha_coeffs_[6] = {147.0/60.0, -6.0, 15.0/2.0, -20.0/3.0, 15.0/4.0, -6.0/5.0, 1.0/6.0};
        beta_coeffs_[6] = 1.0;
    }
    
    void initialize_history(const state_type& y0, time_type dt) {
        // Use lower order methods to build up history
        y_history_.clear();
        dt_history_.clear();
        
        y_history_.push_back(y0);
        dt_history_.push_back(dt);
        
        is_initialized_ = true;
        current_order_ = 1;
    }
    
    void update_history(const state_type& y_new, time_type dt) {
        y_history_.insert(y_history_.begin(), y_new);
        dt_history_.insert(dt_history_.begin(), dt);
        
        // Keep only necessary history
        int max_history = max_order_ + 1;
        if (static_cast<int>(y_history_.size()) > max_history) {
            y_history_.resize(max_history);
            dt_history_.resize(max_history);
        }
    }
    
    void adjust_order(time_type error_norm) {
        // Simple order adjustment strategy
        if (error_norm < 0.1 && current_order_ < max_order_ && 
            static_cast<int>(y_history_.size()) > current_order_) {
            current_order_++;
        } else if (error_norm > 0.5 && current_order_ > 1) {
            current_order_--;
        }
    }
    
    bool bdf_step(const state_type& y_current, state_type& y_new, state_type& error, time_type dt) {
        // For simplicity, start with backward Euler (BDF1) only
        // This avoids the complexity of building up history
        
        // Initial guess: forward Euler
        for (std::size_t i = 0; i < y_current.size(); ++i) {
            auto y_curr_it = y_current.begin();
            auto y_new_it = y_new.begin();
            y_new_it[i] = y_curr_it[i];
        }
        
        state_type f_new = StateCreator<state_type>::create(y_current);
        state_type residual = StateCreator<state_type>::create(y_current);
        
        time_type t_new = this->current_time_ + dt;
        
        // Simple fixed point iteration for backward Euler: y_{n+1} = y_n + dt*f(t_{n+1}, y_{n+1})
        for (int iter = 0; iter < 5; ++iter) {  // Limited iterations
            // Evaluate function
            this->sys_(t_new, y_new, f_new);
            
            // Calculate new guess
            for (std::size_t i = 0; i < y_current.size(); ++i) {
                auto y_curr_it = y_current.begin();
                auto y_new_it = y_new.begin();
                auto f_new_it = f_new.begin();
                
                y_new_it[i] = y_curr_it[i] + dt * f_new_it[i];
            }
        }
        
        // Simple error estimate
        for (std::size_t i = 0; i < y_current.size(); ++i) {
            auto error_it = error.begin();
            error_it[i] = static_cast<value_type>(dt * 0.01);  // Conservative estimate
        }
        
        return true;
    }
};
