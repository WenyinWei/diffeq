#pragma once
#include <core/adaptive_integrator.hpp>
#include <cmath>

// DOP853-style integrator (simplified but more accurate implementation)
// This implements a high-order embedded method similar to DOP853
// For production use, consider using the full 13-stage DOP853 implementation
template<system_state S, can_be_time T = double>
class DOP853Integrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit DOP853Integrator(system_function sys, 
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
            
            dop853_step(state, y_new, error, current_dt);
            
            // Calculate error norm
            time_type err_norm = this->error_norm(error, y_new);
            
            if (err_norm <= 1.0) {
                // Step accepted
                state = y_new;
                this->advance_time(current_dt);
                
                // Conservative step size adjustment for high-order method
                time_type factor = static_cast<time_type>(0.9);
                if (err_norm > 0) {
                    factor *= std::pow(static_cast<time_type>(1.0) / err_norm, static_cast<time_type>(1.0/6.0));
                }
                factor = std::max(static_cast<time_type>(0.2), std::min(factor, static_cast<time_type>(2.0)));
                current_dt = std::max(this->dt_min_, std::min(this->dt_max_, current_dt * factor));
                
                return current_dt;
            } else {
                // Step rejected, reduce step size more aggressively
                time_type factor = static_cast<time_type>(0.5);
                if (err_norm > 1.0) {
                    factor = std::max(static_cast<time_type>(0.1), 
                                    static_cast<time_type>(0.9) * std::pow(static_cast<time_type>(1.0) / err_norm, static_cast<time_type>(1.0/6.0)));
                }
                current_dt = std::max(this->dt_min_, current_dt * factor);
                
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("DOP853: Maximum number of step size reductions exceeded");
    }

private:
    bool dop853_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Simplified DOP853 implementation using a 6-stage method
        // This is not the full DOP853 but provides better accuracy than the previous 4-stage version
        
        // Create temporary states for k calculations
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);
        state_type k5 = StateCreator<state_type>::create(y);
        state_type k6 = StateCreator<state_type>::create(y);
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        // Butcher tableau coefficients for a 6-stage, 5th order method
        // (This is a simplified version, not the full DOP853)
        
        // k1 = f(t, y)
        this->sys_(t, y, k1);
        
        // k2 = f(t + 1/4*dt, y + 1/4*k1*dt)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * static_cast<time_type>(0.25) * k1_it[i];
        }
        this->sys_(t + static_cast<time_type>(0.25) * dt, temp, k2);
        
        // k3 = f(t + 3/8*dt, y + (3/32*k1 + 9/32*k2)*dt)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(3.0/32.0) * k1_it[i] + 
                                        static_cast<time_type>(9.0/32.0) * k2_it[i]);
        }
        this->sys_(t + static_cast<time_type>(3.0/8.0) * dt, temp, k3);
        
        // k4 = f(t + 12/13*dt, y + (1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)*dt)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(1932.0/2197.0) * k1_it[i] - 
                                        static_cast<time_type>(7200.0/2197.0) * k2_it[i] + 
                                        static_cast<time_type>(7296.0/2197.0) * k3_it[i]);
        }
        this->sys_(t + static_cast<time_type>(12.0/13.0) * dt, temp, k4);
        
        // k5 = f(t + dt, y + (439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)*dt)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(439.0/216.0) * k1_it[i] - 
                                        static_cast<time_type>(8.0) * k2_it[i] + 
                                        static_cast<time_type>(3680.0/513.0) * k3_it[i] - 
                                        static_cast<time_type>(845.0/4104.0) * k4_it[i]);
        }
        this->sys_(t + dt, temp, k5);
        
        // k6 = f(t + 1/2*dt, y + (-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)*dt)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (-static_cast<time_type>(8.0/27.0) * k1_it[i] + 
                                        static_cast<time_type>(2.0) * k2_it[i] - 
                                        static_cast<time_type>(3544.0/2565.0) * k3_it[i] + 
                                        static_cast<time_type>(1859.0/4104.0) * k4_it[i] - 
                                        static_cast<time_type>(11.0/40.0) * k5_it[i]);
        }
        this->sys_(t + static_cast<time_type>(0.5) * dt, temp, k6);
        
        // 5th order solution coefficients
        const time_type b1 = static_cast<time_type>(16.0/135.0);
        const time_type b3 = static_cast<time_type>(6656.0/12825.0);
        const time_type b4 = static_cast<time_type>(28561.0/56430.0);
        const time_type b5 = static_cast<time_type>(-9.0/50.0);
        const time_type b6 = static_cast<time_type>(2.0/55.0);
        
        // 4th order solution coefficients (for error estimation)
        const time_type c1 = static_cast<time_type>(25.0/216.0);
        const time_type c3 = static_cast<time_type>(1408.0/2565.0);
        const time_type c4 = static_cast<time_type>(2197.0/4104.0);
        const time_type c5 = static_cast<time_type>(-1.0/5.0);
        
        // Calculate 5th order solution
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto y_new_it = y_new.begin();
            
            y_new_it[i] = y_it[i] + dt * (b1 * k1_it[i] + b3 * k3_it[i] + 
                                         b4 * k4_it[i] + b5 * k5_it[i] + b6 * k6_it[i]);
            
            // Check for NaN or infinite values
            if (!std::isfinite(y_new_it[i])) {
                return false;
            }
        }
        
        // Calculate 4th order solution for error estimation
        state_type y_4th = StateCreator<state_type>::create(y);
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto y_4th_it = y_4th.begin();
            
            y_4th_it[i] = y_it[i] + dt * (c1 * k1_it[i] + c3 * k3_it[i] + 
                                         c4 * k4_it[i] + c5 * k5_it[i]);
        }
        
        // Error = |y_5th - y_4th|
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_new_it = y_new.begin();
            auto y_4th_it = y_4th.begin();
            auto error_it = error.begin();
            
            error_it[i] = std::abs(y_new_it[i] - y_4th_it[i]);
        }
        
        return true;
    }
};
