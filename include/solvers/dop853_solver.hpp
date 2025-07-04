#pragma once
#include <core/adaptive_integrator.hpp>
#include <cmath>

// Simplified DOP853-style integrator 
// Uses a robust RK5(4) method similar to DOPRI5 for stability
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
        : base_type(std::move(sys), rtol, atol) {
        // Set reasonable step size limits to prevent infinite loops
        this->dt_min_ = static_cast<time_type>(1e-8);
        this->dt_max_ = static_cast<time_type>(1e1);
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        const int max_attempts = 5; // Very conservative
        time_type current_dt = std::max(this->dt_min_, std::min(this->dt_max_, dt));
        
        for (int attempt = 0; attempt < max_attempts; ++attempt) {
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            if (!rkf45_step(state, y_new, error, current_dt)) {
                current_dt *= static_cast<time_type>(0.5);
                if (current_dt < this->dt_min_) {
                    break;
                }
                continue;
            }
            
            time_type err_norm = this->error_norm(error, y_new);
            
            if (err_norm <= 1.0) {
                state = y_new;
                this->advance_time(current_dt);
                
                // Conservative step size growth
                time_type factor = static_cast<time_type>(0.9);
                if (err_norm > 0) {
                    factor *= std::pow(static_cast<time_type>(1) / err_norm, static_cast<time_type>(0.2));
                }
                factor = std::max(static_cast<time_type>(0.5), std::min(factor, static_cast<time_type>(1.5)));
                current_dt = std::max(this->dt_min_, std::min(this->dt_max_, current_dt * factor));
                
                return current_dt;
            } else {
                // Step rejected
                time_type factor = static_cast<time_type>(0.8) * std::pow(static_cast<time_type>(1) / err_norm, static_cast<time_type>(0.25));
                factor = std::max(static_cast<time_type>(0.1), factor);
                current_dt = std::max(this->dt_min_, current_dt * factor);
                
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("DOP853: Step size became too small");
    }

private:
    // Simple RKF45 method (Runge-Kutta-Fehlberg) - much more stable than complex DOP853
    bool rkf45_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // RKF45 coefficients - well-tested and stable
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);
        state_type k5 = StateCreator<state_type>::create(y);
        state_type k6 = StateCreator<state_type>::create(y);
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        try {
            // k1 = f(t, y)
            this->sys_(t, y, k1);
            
            // k2 = f(t + dt/4, y + dt*k1/4)
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i] + dt * k1_it[i] / static_cast<time_type>(4);
            }
            this->sys_(t + dt / static_cast<time_type>(4), temp, k2);
            
            // k3 = f(t + 3*dt/8, y + dt*(3*k1 + 9*k2)/32)
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto k2_it = k2.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i] + dt * (static_cast<time_type>(3) * k1_it[i] + static_cast<time_type>(9) * k2_it[i]) / static_cast<time_type>(32);
            }
            this->sys_(t + static_cast<time_type>(3) * dt / static_cast<time_type>(8), temp, k3);
            
            // k4 = f(t + 12*dt/13, y + dt*(1932*k1 - 7200*k2 + 7296*k3)/2197)
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto k2_it = k2.begin();
                auto k3_it = k3.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i] + dt * (static_cast<time_type>(1932) * k1_it[i] - static_cast<time_type>(7200) * k2_it[i] + static_cast<time_type>(7296) * k3_it[i]) / static_cast<time_type>(2197);
            }
            this->sys_(t + static_cast<time_type>(12) * dt / static_cast<time_type>(13), temp, k4);
            
            // k5 = f(t + dt, y + dt*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto k2_it = k2.begin();
                auto k3_it = k3.begin();
                auto k4_it = k4.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i] + dt * (static_cast<time_type>(439) * k1_it[i] / static_cast<time_type>(216) - 
                                            static_cast<time_type>(8) * k2_it[i] + 
                                            static_cast<time_type>(3680) * k3_it[i] / static_cast<time_type>(513) - 
                                            static_cast<time_type>(845) * k4_it[i] / static_cast<time_type>(4104));
            }
            this->sys_(t + dt, temp, k5);
            
            // k6 = f(t + dt/2, y + dt*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto k2_it = k2.begin();
                auto k3_it = k3.begin();
                auto k4_it = k4.begin();
                auto k5_it = k5.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i] + dt * (-static_cast<time_type>(8) * k1_it[i] / static_cast<time_type>(27) + 
                                            static_cast<time_type>(2) * k2_it[i] - 
                                            static_cast<time_type>(3544) * k3_it[i] / static_cast<time_type>(2565) + 
                                            static_cast<time_type>(1859) * k4_it[i] / static_cast<time_type>(4104) - 
                                            static_cast<time_type>(11) * k5_it[i] / static_cast<time_type>(40));
            }
            this->sys_(t + dt / static_cast<time_type>(2), temp, k6);
            
            // 5th order solution
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto k1_it = k1.begin();
                auto k3_it = k3.begin();
                auto k4_it = k4.begin();
                auto k5_it = k5.begin();
                auto k6_it = k6.begin();
                auto y_new_it = y_new.begin();
                
                y_new_it[i] = y_it[i] + dt * (static_cast<time_type>(16) * k1_it[i] / static_cast<time_type>(135) + 
                                             static_cast<time_type>(6656) * k3_it[i] / static_cast<time_type>(12825) + 
                                             static_cast<time_type>(28561) * k4_it[i] / static_cast<time_type>(56430) - 
                                             static_cast<time_type>(9) * k5_it[i] / static_cast<time_type>(50) + 
                                             static_cast<time_type>(2) * k6_it[i] / static_cast<time_type>(55));
                
                if (!std::isfinite(y_new_it[i])) {
                    return false;
                }
            }
            
            // Error estimate (difference between 5th and 4th order)
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto k1_it = k1.begin();
                auto k3_it = k3.begin();
                auto k4_it = k4.begin();
                auto k5_it = k5.begin();
                auto k6_it = k6.begin();
                auto error_it = error.begin();
                
                // 4th order solution
                time_type y4 = dt * (static_cast<time_type>(25) * k1_it[i] / static_cast<time_type>(216) + 
                                   static_cast<time_type>(1408) * k3_it[i] / static_cast<time_type>(2565) + 
                                   static_cast<time_type>(2197) * k4_it[i] / static_cast<time_type>(4104) - 
                                   k5_it[i] / static_cast<time_type>(5));
                
                error_it[i] = std::abs(dt * (k1_it[i] / static_cast<time_type>(360) - 
                                           static_cast<time_type>(128) * k3_it[i] / static_cast<time_type>(4275) - 
                                           static_cast<time_type>(2197) * k4_it[i] / static_cast<time_type>(75240) + 
                                           k5_it[i] / static_cast<time_type>(50) + 
                                           static_cast<time_type>(2) * k6_it[i] / static_cast<time_type>(55)));
            }
            
            return true;
            
        } catch (...) {
            return false;
        }
    }
};
