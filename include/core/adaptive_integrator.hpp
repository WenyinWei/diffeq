#pragma once
#include <functional>
#include <concepts>
#include <iterator>
#include <type_traits>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/state_creator.hpp>

// Abstract adaptive integrator with error control
template<system_state S, can_be_time T = double>
class AdaptiveIntegrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit AdaptiveIntegrator(system_function sys, 
                               time_type rtol = static_cast<time_type>(1e-6),
                               time_type atol = static_cast<time_type>(1e-9))
        : base_type(std::move(sys)), rtol_(rtol), atol_(atol),
          dt_min_(static_cast<time_type>(1e-12)), dt_max_(static_cast<time_type>(1e2)),
          safety_factor_(static_cast<time_type>(0.9)) {}

    // Override integrate to use adaptive stepping
    void integrate(state_type& state, time_type dt, time_type end_time) override {
        time_type current_dt = dt;
        
        while (this->current_time_ < end_time) {
            if (this->current_time_ + current_dt > end_time) {
                current_dt = end_time - this->current_time_;
            }
            
            current_dt = adaptive_step(state, current_dt);
            
            if (current_dt < dt_min_) {
                throw std::runtime_error("Step size became too small in adaptive integration");
            }
        }
    }

    // Pure virtual adaptive step - derived classes implement this
    virtual time_type adaptive_step(state_type& state, time_type dt) = 0;

    // Setters for tolerances
    void set_tolerances(time_type rtol, time_type atol) {
        rtol_ = rtol;
        atol_ = atol;
    }

    void set_step_limits(time_type dt_min, time_type dt_max) {
        dt_min_ = dt_min;
        dt_max_ = dt_max;
    }

protected:
    time_type rtol_, atol_;           // Relative and absolute tolerances
    time_type dt_min_, dt_max_;       // Step size limits
    time_type safety_factor_;        // Safety factor for step size adjustment

    // Calculate error tolerance for each component
    time_type calculate_tolerance(value_type y_val) const {
        return atol_ + rtol_ * std::abs(y_val);
    }

    // Calculate error norm using SciPy-style L2 norm
    time_type error_norm(const state_type& error, const state_type& y) const {
        time_type norm_squared = static_cast<time_type>(0);
        std::size_t n = 0;
        
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto err_it = error.begin();
            time_type scale = atol_ + std::abs(y_it[i]) * rtol_;
            time_type scaled_error = err_it[i] / scale;
            norm_squared += scaled_error * scaled_error;
            ++n;
        }
        
        if (n == 0) return static_cast<time_type>(0);
        return std::sqrt(norm_squared / n);
    }

    // SciPy-style error norm calculation using max of current and new state
    time_type error_norm_scipy_style(const state_type& error, const state_type& y_old, const state_type& y_new) const {
        time_type norm_squared = static_cast<time_type>(0);
        std::size_t n = 0;
        
        for (std::size_t i = 0; i < y_old.size(); ++i) {
            auto y_old_it = y_old.begin();
            auto y_new_it = y_new.begin();
            auto err_it = error.begin();
            
            // SciPy uses: scale = atol + max(abs(y), abs(y_new)) * rtol
            time_type scale = atol_ + std::max(std::abs(y_old_it[i]), std::abs(y_new_it[i])) * rtol_;
            time_type scaled_error = err_it[i] / scale;
            norm_squared += scaled_error * scaled_error;
            ++n;
        }
        
        if (n == 0) return static_cast<time_type>(0);
        return std::sqrt(norm_squared / n);
    }

    // Suggest new step size based on error
    time_type suggest_step_size(time_type current_dt, time_type error_norm, int order) const {
        if (error_norm == 0) {
            return std::min(current_dt * static_cast<time_type>(2), dt_max_);
        }
        
        time_type factor = safety_factor_ * std::pow(static_cast<time_type>(1) / error_norm, 
                                                   static_cast<time_type>(1) / order);
        factor = std::max(static_cast<time_type>(0.1), std::min(factor, static_cast<time_type>(5.0)));
        
        return std::max(dt_min_, std::min(dt_max_, current_dt * factor));
    }
};
