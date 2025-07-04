#pragma once
#include <core/adaptive_integrator.hpp>
#include <cmath>

// RK45 (Dormand-Prince) integrator
// Fourth-order method with fifth-order error estimation
// This is the most popular adaptive RK method
template<system_state S, can_be_time T = double>
class RK45Integrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit RK45Integrator(system_function sys, 
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
            
            rk45_step(state, y_new, error, current_dt);
            
            // Calculate error norm using SciPy-style scaling
            time_type err_norm = this->error_norm_scipy_style(error, state, y_new);
            
            if (err_norm <= 1.0) {
                // Step accepted
                state = y_new;
                this->advance_time(current_dt);
                
                // SciPy-style step size control
                // For RK45: error_estimator_order = 4, so exponent = -1/(4+1) = -1/5
                time_type safety = static_cast<time_type>(0.9);
                time_type min_factor = static_cast<time_type>(0.2);
                time_type max_factor = static_cast<time_type>(10.0);
                time_type error_exponent = static_cast<time_type>(-1.0 / 5.0);  // -(1/(error_estimator_order + 1))
                
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
                time_type error_exponent = static_cast<time_type>(-1.0 / 5.0);  // -(1/(error_estimator_order + 1))
                
                time_type factor = std::max(min_factor, 
                                           safety * std::pow(err_norm, error_exponent));
                current_dt = std::max(this->dt_min_, current_dt * factor);
                
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("RK45: Maximum number of step size reductions exceeded");
    }

private:
    void rk45_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Dormand-Prince coefficients matching SciPy's RK45
        // C = [0, 1/5, 3/10, 4/5, 8/9, 1]
        // A = (see SciPy implementation)
        // B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84] (5th order)
        // E = [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40] (error)
        
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);
        state_type k5 = StateCreator<state_type>::create(y);
        state_type k6 = StateCreator<state_type>::create(y);
        state_type k7 = StateCreator<state_type>::create(y);  // For error estimation
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        // k1 = f(t, y)
        this->sys_(t, y, k1);
        
        // k2 = f(t + dt/5, y + dt*k1/5)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * k1_it[i] / static_cast<time_type>(5);
        }
        this->sys_(t + dt / static_cast<time_type>(5), temp, k2);
        
        // k3 = f(t + 3*dt/10, y + dt*(3*k1/40 + 9*k2/40))
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(3) * k1_it[i] / static_cast<time_type>(40) + 
                                        static_cast<time_type>(9) * k2_it[i] / static_cast<time_type>(40));
        }
        this->sys_(t + static_cast<time_type>(3) * dt / static_cast<time_type>(10), temp, k3);
        
        // k4 = f(t + 4*dt/5, y + dt*(44*k1/45 - 56*k2/15 + 32*k3/9))
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(44) * k1_it[i] / static_cast<time_type>(45) + 
                                        static_cast<time_type>(-56) * k2_it[i] / static_cast<time_type>(15) + 
                                        static_cast<time_type>(32) * k3_it[i] / static_cast<time_type>(9));
        }
        this->sys_(t + static_cast<time_type>(4) * dt / static_cast<time_type>(5), temp, k4);
        
        // k5 = f(t + 8*dt/9, y + dt*(19372*k1/6561 - 25360*k2/2187 + 64448*k3/6561 - 212*k4/729))
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(19372) * k1_it[i] / static_cast<time_type>(6561) + 
                                        static_cast<time_type>(-25360) * k2_it[i] / static_cast<time_type>(2187) + 
                                        static_cast<time_type>(64448) * k3_it[i] / static_cast<time_type>(6561) + 
                                        static_cast<time_type>(-212) * k4_it[i] / static_cast<time_type>(729));
        }
        this->sys_(t + static_cast<time_type>(8) * dt / static_cast<time_type>(9), temp, k5);
        
        // k6 = f(t + dt, y + dt*(9017*k1/3168 - 355*k2/33 + 46732*k3/5247 + 49*k4/176 - 5103*k5/18656))
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto temp_it = temp.begin();
            temp_it[i] = y_it[i] + dt * (static_cast<time_type>(9017) * k1_it[i] / static_cast<time_type>(3168) + 
                                        static_cast<time_type>(-355) * k2_it[i] / static_cast<time_type>(33) + 
                                        static_cast<time_type>(46732) * k3_it[i] / static_cast<time_type>(5247) + 
                                        static_cast<time_type>(49) * k4_it[i] / static_cast<time_type>(176) + 
                                        static_cast<time_type>(-5103) * k5_it[i] / static_cast<time_type>(18656));
        }
        this->sys_(t + dt, temp, k6);
        
        // 5th order solution: y_new = y + dt*(35*k1/384 + 500*k3/1113 + 125*k4/192 - 2187*k5/6784 + 11*k6/84)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto y_new_it = y_new.begin();
            
            y_new_it[i] = y_it[i] + dt * (static_cast<time_type>(35) * k1_it[i] / static_cast<time_type>(384) + 
                                         static_cast<time_type>(500) * k3_it[i] / static_cast<time_type>(1113) + 
                                         static_cast<time_type>(125) * k4_it[i] / static_cast<time_type>(192) + 
                                         static_cast<time_type>(-2187) * k5_it[i] / static_cast<time_type>(6784) + 
                                         static_cast<time_type>(11) * k6_it[i] / static_cast<time_type>(84));
        }
        
        // k7 = f(t + dt, y_new) - needed for error estimation
        this->sys_(t + dt, y_new, k7);
        
        // Error estimate using E = [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto k7_it = k7.begin();
            auto error_it = error.begin();
            
            error_it[i] = dt * (static_cast<time_type>(-71) * k1_it[i] / static_cast<time_type>(57600) + 
                               static_cast<time_type>(71) * k3_it[i] / static_cast<time_type>(16695) + 
                               static_cast<time_type>(-71) * k4_it[i] / static_cast<time_type>(1920) + 
                               static_cast<time_type>(17253) * k5_it[i] / static_cast<time_type>(339200) + 
                               static_cast<time_type>(-22) * k6_it[i] / static_cast<time_type>(525) + 
                               k7_it[i] / static_cast<time_type>(40));
        }
    }
};
