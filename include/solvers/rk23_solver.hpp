#pragma once
#include <core/adaptive_integrator.hpp>

// RK23 (Bogacki-Shampine) integrator
// Second-order method with third-order error estimation
template<system_state S, can_be_time T = double>
class RK23Integrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit RK23Integrator(system_function sys, 
                           time_type rtol = static_cast<time_type>(1e-6),
                           time_type atol = static_cast<time_type>(1e-9))
        : base_type(std::move(sys), rtol, atol) {}

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        const int max_attempts = 10;
        time_type current_dt = dt;
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            rk23_step(state, y_new, error, current_dt);
            
            // Calculate error norm using SciPy-style scaling
            time_type err_norm = this->error_norm_scipy_style(error, state, y_new);
            
            if (err_norm <= 1.0) {
                // Step accepted
                state = y_new;
                this->advance_time(current_dt);
                
                // SciPy-style step size control
                // For RK23: error_estimator_order = 2, so exponent = -1/(2+1) = -1/3
                time_type safety = static_cast<time_type>(0.9);
                time_type min_factor = static_cast<time_type>(0.2);
                time_type max_factor = static_cast<time_type>(10.0);
                time_type error_exponent = static_cast<time_type>(-1.0 / 3.0);  // -(1/(error_estimator_order + 1))
                
                time_type factor;
                if (err_norm == 0) {
                    factor = max_factor;
                } else {
                    factor = std::min(max_factor, safety * std::pow(err_norm, error_exponent));
                }
                
                // If step was rejected, limit growth
                factor = std::max(min_factor, std::min(max_factor, factor));
                current_dt = std::max(this->dt_min_, std::min(this->dt_max_, current_dt * factor));
                
                return current_dt;
            } else {
                // Step rejected, reduce step size
                time_type safety = static_cast<time_type>(0.9);
                time_type min_factor = static_cast<time_type>(0.2);
                time_type error_exponent = static_cast<time_type>(-1.0 / 3.0);  // -(1/(error_estimator_order + 1))
                
                time_type factor = std::max(min_factor, 
                                           safety * std::pow(err_norm, error_exponent));
                current_dt = std::max(this->dt_min_, current_dt * factor);
                
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("RK23: Maximum number of step size reductions exceeded");
    }

private:
    void rk23_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Bogacki-Shampine coefficients matching SciPy's RK23
        // C = [0, 1/2, 3/4]
        // A = [[0, 0, 0], [1/2, 0, 0], [0, 3/4, 0]]
        // B = [2/9, 1/3, 4/9] (3rd order)
        // E = [5/72, -1/12, -1/9, 1/8] (error estimate)
        
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);  // For error estimation
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        // k1 = f(t, y)
        this->sys_(t, y, k1);
        
        // k2 = f(t + dt/2, y + dt*k1/2)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * k1_it[i] / static_cast<time_type>(2);
        }
        this->sys_(t + dt / static_cast<time_type>(2), temp, k2);
        
        // k3 = f(t + 3*dt/4, y + 3*dt*k2/4)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * static_cast<time_type>(3) * k2_it[i] / static_cast<time_type>(4);
        }
        this->sys_(t + static_cast<time_type>(3) * dt / static_cast<time_type>(4), temp, k3);
        
        // 3rd order solution: y_new = y + dt*(2*k1/9 + k2/3 + 4*k3/9)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto y_new_it = y_new.begin();
            
            y_new_it[i] = y_it[i] + dt * (static_cast<time_type>(2) * k1_it[i] / static_cast<time_type>(9) + 
                                         k2_it[i] / static_cast<time_type>(3) + 
                                         static_cast<time_type>(4) * k3_it[i] / static_cast<time_type>(9));
        }
        
        // k4 = f(t + dt, y_new) - needed for error estimation
        this->sys_(t + dt, y_new, k4);
        
        // Error estimate using E = [5/72, -1/12, -1/9, 1/8]
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto error_it = error.begin();
            
            error_it[i] = dt * (static_cast<time_type>(5) * k1_it[i] / static_cast<time_type>(72) + 
                               static_cast<time_type>(-1) * k2_it[i] / static_cast<time_type>(12) + 
                               static_cast<time_type>(-1) * k3_it[i] / static_cast<time_type>(9) + 
                               k4_it[i] / static_cast<time_type>(8));
        }
    }
};
