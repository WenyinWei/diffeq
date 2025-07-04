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
            // Try a step
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            
            rk45_step(state, y_new, error, current_dt);
            
            // Calculate error norm
            time_type err_norm = this->error_norm(error, y_new);
            
            // More lenient error control for debugging
            if (err_norm <= 10.0 || current_dt <= this->dt_min_) {
                // Step accepted
                state = y_new;
                this->advance_time(current_dt);
                
                // Suggest next step size (more conservative)
                if (err_norm <= 1.0) {
                    current_dt = this->suggest_step_size(current_dt, err_norm, 4);
                } else {
                    current_dt = std::min(current_dt, this->dt_max_);
                }
                return current_dt;
            } else {
                // Step rejected, reduce step size
                current_dt *= static_cast<time_type>(0.5);
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("RK45: Maximum number of step size reductions exceeded");
    }

private:
    void rk45_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Dormand-Prince coefficients
        const time_type a21 = static_cast<time_type>(1.0/5.0);
        const time_type a31 = static_cast<time_type>(3.0/40.0);
        const time_type a32 = static_cast<time_type>(9.0/40.0);
        const time_type a41 = static_cast<time_type>(44.0/45.0);
        const time_type a42 = static_cast<time_type>(-56.0/15.0);
        const time_type a43 = static_cast<time_type>(32.0/9.0);
        const time_type a51 = static_cast<time_type>(19372.0/6561.0);
        const time_type a52 = static_cast<time_type>(-25360.0/2187.0);
        const time_type a53 = static_cast<time_type>(64448.0/6561.0);
        const time_type a54 = static_cast<time_type>(-212.0/729.0);
        const time_type a61 = static_cast<time_type>(9017.0/3168.0);
        const time_type a62 = static_cast<time_type>(-355.0/33.0);
        const time_type a63 = static_cast<time_type>(46732.0/5247.0);
        const time_type a64 = static_cast<time_type>(49.0/176.0);
        const time_type a65 = static_cast<time_type>(-5103.0/18656.0);
        
        // Fourth-order solution coefficients
        const time_type b1 = static_cast<time_type>(35.0/384.0);
        const time_type b3 = static_cast<time_type>(500.0/1113.0);
        const time_type b4 = static_cast<time_type>(125.0/192.0);
        const time_type b5 = static_cast<time_type>(-2187.0/6784.0);
        const time_type b6 = static_cast<time_type>(11.0/84.0);
        
        // Fifth-order solution coefficients (for error estimation)
        const time_type c1 = static_cast<time_type>(5179.0/57600.0);
        const time_type c3 = static_cast<time_type>(7571.0/16695.0);
        const time_type c4 = static_cast<time_type>(393.0/640.0);
        const time_type c5 = static_cast<time_type>(-92097.0/339200.0);
        const time_type c6 = static_cast<time_type>(187.0/2100.0);
        const time_type c7 = static_cast<time_type>(1.0/40.0);
        
        // Temporary states
        state_type k1 = StateCreator<state_type>::create(y);
        state_type k2 = StateCreator<state_type>::create(y);
        state_type k3 = StateCreator<state_type>::create(y);
        state_type k4 = StateCreator<state_type>::create(y);
        state_type k5 = StateCreator<state_type>::create(y);
        state_type k6 = StateCreator<state_type>::create(y);
        state_type k7 = StateCreator<state_type>::create(y);
        state_type temp = StateCreator<state_type>::create(y);
        
        // k1 = f(t, y)
        this->sys_(this->current_time_, y, k1);
        
        // k2 = f(t + dt/5, y + dt*k1/5)
        for (std::size_t i = 0; i < y.size(); ++i) {
            temp[i] = y[i] + dt * a21 * k1[i];
        }
        this->sys_(this->current_time_ + dt / 5, temp, k2);
        
        // k3 = f(t + 3*dt/10, y + dt*(3*k1 + 9*k2)/40)
        for (std::size_t i = 0; i < y.size(); ++i) {
            temp[i] = y[i] + dt * (a31 * k1[i] + a32 * k2[i]);
        }
        this->sys_(this->current_time_ + dt * 3 / 10, temp, k3);
        
        // k4 = f(t + 4*dt/5, y + dt*(44*k1 - 56*k2 + 32*k3)/45)
        for (std::size_t i = 0; i < y.size(); ++i) {
            temp[i] = y[i] + dt * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        this->sys_(this->current_time_ + dt * 4 / 5, temp, k4);
        
        // k5 = f(t + 8*dt/9, y + dt*(19372*k1 - 25360*k2 + 64448*k3 - 212*k4)/6561)
        for (std::size_t i = 0; i < y.size(); ++i) {
            temp[i] = y[i] + dt * (a51 * k1[i] + a52 * k2[i] + 
                                  a53 * k3[i] + a54 * k4[i]);
        }
        this->sys_(this->current_time_ + dt * 8 / 9, temp, k5);
        
        // k6 = f(t + dt, y + dt*(9017*k1 - 355*k2 + 46732*k3 + 49*k4 - 5103*k5)/3168)
        for (std::size_t i = 0; i < y.size(); ++i) {
            temp[i] = y[i] + dt * (a61 * k1[i] + a62 * k2[i] + 
                                  a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        this->sys_(this->current_time_ + dt, temp, k6);
        
        // Fourth-order solution
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
        }
        
        // k7 = f(t + dt, y_new) for fifth-order estimate
        this->sys_(this->current_time_ + dt, y_new, k7);
        
        // Simplified error estimate: use difference between 4th and 5th order solutions
        // Calculate 5th order solution
        state_type y_5th = StateCreator<state_type>::create(y);
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto k7_it = k7.begin();
            auto y_5th_it = y_5th.begin();
            
            y_5th_it[i] = y_it[i] + dt * (c1 * k1_it[i] + c3 * k3_it[i] + c4 * k4_it[i] + 
                                         c5 * k5_it[i] + c6 * k6_it[i] + c7 * k7_it[i]);
        }
        
        // Error = |y_5th - y_4th|
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_new_it = y_new.begin();
            auto y_5th_it = y_5th.begin();
            auto error_it = error.begin();
            
            error_it[i] = std::abs(y_5th_it[i] - y_new_it[i]);
        }
    }
};
