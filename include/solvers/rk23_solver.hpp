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
            // Try a step
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            rk23_step(state, y_new, error, current_dt);
            
            // Calculate error norm
            time_type err_norm = this->error_norm(error, state);
            
            if (err_norm <= 1.0) {
                // Step accepted
                state = y_new;
                this->advance_time(current_dt);
                
                // Suggest next step size
                current_dt = this->suggest_step_size(current_dt, err_norm, 3);
                return current_dt;
            } else {
                // Step rejected, reduce step size
                current_dt = this->suggest_step_size(current_dt, err_norm, 3);
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("RK23: Maximum number of step size reductions exceeded");
    }

private:
    void rk23_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Bogacki-Shampine coefficients
        const time_type a21 = static_cast<time_type>(1.0/2.0);
        const time_type a32 = static_cast<time_type>(3.0/4.0);
        const time_type a43 = static_cast<time_type>(1.0/3.0);
        
        const time_type b1 = static_cast<time_type>(2.0/9.0);
        const time_type b2 = static_cast<time_type>(1.0/3.0);
        const time_type b3 = static_cast<time_type>(4.0/9.0);
        
        const time_type c1 = static_cast<time_type>(7.0/24.0);
        const time_type c2 = static_cast<time_type>(1.0/4.0);
        const time_type c3 = static_cast<time_type>(1.0/3.0);
        const time_type c4 = static_cast<time_type>(1.0/8.0);
        
        // Temporary states
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);
        state_type temp = StateCreator<state_type>::create(y);
        
        // k1 = f(t, y)
        this->sys_(this->current_time_, y, k1);
        
        // k2 = f(t + dt/2, y + dt*k1/2)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * a21 * k1_it[i];
        }
        this->sys_(this->current_time_ + dt * a21, temp, k2);
        
        // k3 = f(t + 3*dt/4, y + 3*dt*k2/4)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * a32 * k2_it[i];
        }
        this->sys_(this->current_time_ + dt * a32, temp, k3);
        
        // k4 = f(t + dt, y + dt*(2*k1 + 3*k2 + 4*k3)/9)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (b1 * k1_it[i] + b2 * k2_it[i] + b3 * k3_it[i]);
        }
        this->sys_(this->current_time_ + dt, temp, k4);
        
        // Second-order solution: y_new = y + dt*(7*k1 + 6*k2 + 8*k3 + 3*k4)/24
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto y_new_it = y_new.begin();
            
            y_new_it[i] = y_it[i] + dt * (c1 * k1_it[i] + c2 * k2_it[i] + 
                                         c3 * k3_it[i] + c4 * k4_it[i]);
        }
        
        // Error estimate: difference between 2nd and 3rd order methods
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto y_new_it = y_new.begin();
            auto error_it = error.begin();
            
            // Third-order solution
            value_type y3 = y_it[i] + dt * (b1 * k1_it[i] + b2 * k2_it[i] + b3 * k3_it[i]);
            error_it[i] = y_new_it[i] - y3;
        }
    }
};
