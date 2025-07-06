#pragma once
#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <stdexcept>

namespace diffeq {

/**
 * @brief RK23 (Bogacki-Shampine) adaptive integrator
 * 
 * Second-order method with third-order error estimation.
 * Good for problems that don't require high accuracy.
 * Lower computational cost than RK45.
 * 
 * Order: 2(3) - 2nd order method with 3rd order error estimation
 * Stages: 4
 * Adaptive: Yes
 */
template<system_state S>
class RK23Integrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
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
            
            // Calculate error norm
            time_type err_norm = this->error_norm(error, y_new);
            
            if (err_norm <= 1.0) {
                // Accept step
                state = y_new;
                this->advance_time(current_dt);
                
                // Suggest next step size
                time_type next_dt = this->suggest_step_size(current_dt, err_norm, 3);
                return std::max(this->dt_min_, std::min(this->dt_max_, next_dt));
            } else {
                // Reject step and reduce step size
                current_dt *= std::max(this->safety_factor_ * std::pow(err_norm, -1.0/3.0), 
                                     static_cast<time_type>(0.1));
                current_dt = std::max(current_dt, this->dt_min_);
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
            temp_it[i] = y_it[i] + static_cast<time_type>(3) * dt * k2_it[i] / static_cast<time_type>(4);
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
            
            error_it[i] = dt * (static_cast<time_type>(5) * k1_it[i] / static_cast<time_type>(72) -
                               k2_it[i] / static_cast<time_type>(12) -
                               k3_it[i] / static_cast<time_type>(9) +
                               k4_it[i] / static_cast<time_type>(8));
        }
    }
};

} // namespace diffeq
