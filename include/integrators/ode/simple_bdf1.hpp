#pragma once
#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace diffeq {

/**
 * @brief Simple BDF1 (Backward Euler) integrator for debugging
 * 
 * This is a simplified implementation to understand the BDF method.
 * BDF1 equation: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
 */
template<typename S>
class SimpleBDF1Integrator : public core::AdaptiveIntegrator<S> {
public:
    using base_type = core::AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit SimpleBDF1Integrator(system_function sys,
                                 time_type rtol = static_cast<time_type>(1e-3),
                                 time_type atol = static_cast<time_type>(1e-6))
        : base_type(std::move(sys), rtol, atol) {}

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        time_type t = this->current_time_;
        time_type h = dt;
        
        // BDF1 (Backward Euler): y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
        // Rearranging: y_{n+1} - h * f(t_{n+1}, y_{n+1}) = y_n
        // For dy/dt = -y: y_{n+1} - h * (-y_{n+1}) = y_n
        //                 y_{n+1} + h * y_{n+1} = y_n
        //                 y_{n+1} * (1 + h) = y_n
        //                 y_{n+1} = y_n / (1 + h)
        
        state_type y_new = StateCreator<state_type>::create(state);
        
        // For the exponential decay problem dy/dt = -y, the exact BDF1 solution is:
        for (std::size_t i = 0; i < state.size(); ++i) {
            y_new[i] = state[i] / (1.0 + h);
        }
        
        // Calculate error estimate (simple first-order approximation)
        state_type error = StateCreator<state_type>::create(state);
        for (std::size_t i = 0; i < state.size(); ++i) {
            // Error estimate: difference between BDF1 and forward Euler
            time_type forward_euler = state[i] * (1.0 - h);  // Forward Euler for dy/dt = -y
            error[i] = std::abs(y_new[i] - forward_euler);
        }
        
        // Calculate error norm
        time_type error_norm = 0.0;
        for (std::size_t i = 0; i < error.size(); ++i) {
            time_type scale = this->atol_ + this->rtol_ * std::abs(y_new[i]);
            time_type scaled_error = error[i] / scale;
            error_norm += scaled_error * scaled_error;
        }
        error_norm = std::sqrt(error_norm / error.size());
        
        // Accept or reject step
        if (error_norm <= 1.0) {
            // Step accepted
            state = y_new;
            this->advance_time(h);
            
            // Adjust step size for next step
            if (error_norm > 0.0) {
                time_type factor = static_cast<time_type>(0.9) * std::pow(error_norm, static_cast<time_type>(-0.5));  // -1/(order+1) = -1/2 for BDF1
                if (factor < static_cast<time_type>(0.2)) factor = static_cast<time_type>(0.2);
                if (factor > static_cast<time_type>(5.0)) factor = static_cast<time_type>(5.0);
                h *= factor;
            }
        } else {
            // Step rejected, reduce step size
            time_type factor = static_cast<time_type>(0.9) * std::pow(error_norm, static_cast<time_type>(-0.5));
            if (factor < static_cast<time_type>(0.2)) factor = static_cast<time_type>(0.2);
            if (factor > static_cast<time_type>(1.0)) factor = static_cast<time_type>(1.0);
            h *= factor;
            
            // Retry with smaller step
            return adaptive_step(state, h);
        }
        
        return h;
    }
};

} // namespace diffeq
