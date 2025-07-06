#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <stdexcept>

namespace diffeq::integrators::ode {

/**
 * @brief RK45 (Runge-Kutta-Fehlberg 4(5)) adaptive integrator
 * 
 * Fifth-order method with embedded 4th order error estimation.
 * Popular adaptive ODE solver used in many scientific libraries.
 * 
 * Order: 5 (with 4th order error estimation)
 * Stability: Good for non-stiff problems
 * Usage: General-purpose adaptive ODE integration
 */
template<system_state S>
class RK45Integrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
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
        // Create temporary states for RK45 calculations
        state_type k1 = StateCreator<state_type>::create(state);
        state_type k2 = StateCreator<state_type>::create(state);
        state_type k3 = StateCreator<state_type>::create(state);
        state_type k4 = StateCreator<state_type>::create(state);
        state_type k5 = StateCreator<state_type>::create(state);
        state_type k6 = StateCreator<state_type>::create(state);
        state_type temp_state = StateCreator<state_type>::create(state);
        state_type y_new = StateCreator<state_type>::create(state);
        state_type error = StateCreator<state_type>::create(state);

        // RK45 coefficients (Fehlberg coefficients)
        constexpr time_type a2 = static_cast<time_type>(1.0/5.0);
        constexpr time_type a3 = static_cast<time_type>(3.0/10.0);
        constexpr time_type a4 = static_cast<time_type>(4.0/5.0);
        constexpr time_type a5 = static_cast<time_type>(8.0/9.0);
        
        constexpr time_type b21 = static_cast<time_type>(1.0/5.0);
        constexpr time_type b31 = static_cast<time_type>(3.0/40.0);
        constexpr time_type b32 = static_cast<time_type>(9.0/40.0);
        constexpr time_type b41 = static_cast<time_type>(44.0/45.0);
        constexpr time_type b42 = static_cast<time_type>(-56.0/15.0);
        constexpr time_type b43 = static_cast<time_type>(32.0/9.0);
        constexpr time_type b51 = static_cast<time_type>(19372.0/6561.0);
        constexpr time_type b52 = static_cast<time_type>(-25360.0/2187.0);
        constexpr time_type b53 = static_cast<time_type>(64448.0/6561.0);
        constexpr time_type b54 = static_cast<time_type>(-212.0/729.0);
        constexpr time_type b61 = static_cast<time_type>(9017.0/3168.0);
        constexpr time_type b62 = static_cast<time_type>(-355.0/33.0);
        constexpr time_type b63 = static_cast<time_type>(46732.0/5247.0);
        constexpr time_type b64 = static_cast<time_type>(49.0/176.0);
        constexpr time_type b65 = static_cast<time_type>(-5103.0/18656.0);

        // 5th order solution coefficients
        constexpr time_type c1 = static_cast<time_type>(35.0/384.0);
        constexpr time_type c3 = static_cast<time_type>(500.0/1113.0);
        constexpr time_type c4 = static_cast<time_type>(125.0/192.0);
        constexpr time_type c5 = static_cast<time_type>(-2187.0/6784.0);
        constexpr time_type c6 = static_cast<time_type>(11.0/84.0);

        // 4th order solution coefficients (for error estimation)
        constexpr time_type c1_4 = static_cast<time_type>(5179.0/57600.0);
        constexpr time_type c3_4 = static_cast<time_type>(7571.0/16695.0);
        constexpr time_type c4_4 = static_cast<time_type>(393.0/640.0);
        constexpr time_type c5_4 = static_cast<time_type>(-92097.0/339200.0);
        constexpr time_type c6_4 = static_cast<time_type>(187.0/2100.0);
        constexpr time_type c7_4 = static_cast<time_type>(1.0/40.0);

        // k1 = f(t, y)
        this->sys_(this->current_time_, state, k1);
        
        // k2 = f(t + a2*dt, y + dt*(b21*k1))
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp_state.begin();
            temp_it[i] = state_it[i] + dt * b21 * k1_it[i];
        }
        this->sys_(this->current_time_ + a2 * dt, temp_state, k2);
        
        // k3 = f(t + a3*dt, y + dt*(b31*k1 + b32*k2))
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp_state.begin();
            temp_it[i] = state_it[i] + dt * (b31 * k1_it[i] + b32 * k2_it[i]);
        }
        this->sys_(this->current_time_ + a3 * dt, temp_state, k3);
        
        // k4 = f(t + a4*dt, y + dt*(b41*k1 + b42*k2 + b43*k3))
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto temp_it = temp_state.begin();
            temp_it[i] = state_it[i] + dt * (b41 * k1_it[i] + b42 * k2_it[i] + b43 * k3_it[i]);
        }
        this->sys_(this->current_time_ + a4 * dt, temp_state, k4);
        
        // k5 = f(t + a5*dt, y + dt*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto temp_it = temp_state.begin();
            temp_it[i] = state_it[i] + dt * (b51 * k1_it[i] + b52 * k2_it[i] + b53 * k3_it[i] + b54 * k4_it[i]);
        }
        this->sys_(this->current_time_ + a5 * dt, temp_state, k5);
        
        // k6 = f(t + dt, y + dt*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto temp_it = temp_state.begin();
            temp_it[i] = state_it[i] + dt * (b61 * k1_it[i] + b62 * k2_it[i] + b63 * k3_it[i] + b64 * k4_it[i] + b65 * k5_it[i]);
        }
        this->sys_(this->current_time_ + dt, temp_state, k6);
        
        // 5th order solution: y_new = y + dt*(c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto y_new_it = y_new.begin();
            y_new_it[i] = state_it[i] + dt * (c1 * k1_it[i] + c3 * k3_it[i] + c4 * k4_it[i] + c5 * k5_it[i] + c6 * k6_it[i]);
        }
        
        // 4th order solution for error estimation
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            auto k5_it = k5.begin();
            auto k6_it = k6.begin();
            auto error_it = error.begin();
            error_it[i] = dt * ((c1 - c1_4) * k1_it[i] + (c3 - c3_4) * k3_it[i] + (c4 - c4_4) * k4_it[i] + 
                               (c5 - c5_4) * k5_it[i] + (c6 - c6_4) * k6_it[i]);
        }
        
        // Calculate error norm
        time_type error_norm = this->error_norm_scipy_style(error, state, y_new);
        
        // Accept or reject step
        if (error_norm <= static_cast<time_type>(1.0)) {
            // Accept step
            state = y_new;
            this->advance_time(dt);
            
            // Suggest next step size
            time_type new_dt = this->suggest_step_size(dt, error_norm, 5);
            return new_dt;
        } else {
            // Reject step, suggest smaller step size
            time_type new_dt = this->suggest_step_size(dt, error_norm, 5);
            return new_dt;
        }
    }
};

} // namespace diffeq::integrators::ode
