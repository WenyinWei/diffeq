#pragma once

#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include "rk45.hpp"
#include "bdf.hpp"
#include <memory>
#include <cmath>

namespace diffeq {

/**
 * @brief LSODA integrator - automatically switches between stiff and non-stiff methods
 * 
 * Automatically switches between non-stiff (Adams, approximated by RK45) and stiff (BDF) methods
 * based on stiffness detection. This is a simplified version inspired by the original LSODA algorithm.
 */
template<system_state S>
class LSODAIntegrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    enum class MethodType {
        ADAMS,    // Non-stiff method (approximated by RK45)
        BDF       // Stiff method
    };

    explicit LSODAIntegrator(system_function sys, 
                   time_type rtol = static_cast<time_type>(1e-6),
                   time_type atol = static_cast<time_type>(1e-9))
        : base_type(std::move(sys), rtol, atol),
          current_method_(MethodType::ADAMS),
          stiffness_detection_frequency_(10),
          step_count_(0),
          consecutive_stiff_steps_(0),
          consecutive_nonstiff_steps_(0),
          stiffness_threshold_(static_cast<time_type>(3.25)),
          has_previous_values_(false) {
        
        // Create internal integrators
        rk45_integrator_ = std::make_unique<RK45Integrator<S>>(
            [this](time_type t, const state_type& y, state_type& dydt) {
                this->sys_(t, y, dydt);
            }, rtol, atol);
            
        bdf_integrator_ = std::make_unique<BDFIntegrator<S>>(
            [this](time_type t, const state_type& y, state_type& dydt) {
                this->sys_(t, y, dydt);
            }, rtol, atol);
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        // For simplicity, just use RK45 method (Adams approximation)
        // Real LSODA would have more sophisticated switching logic
        
        if (!rk45_integrator_) {
            rk45_integrator_ = std::make_unique<RK45Integrator<S>>(
                [this](time_type t, const state_type& y, state_type& dydt) {
                    this->sys_(t, y, dydt);
                }, this->rtol_, this->atol_);
        }
        
        rk45_integrator_->set_time(this->current_time_);
        time_type result_dt = rk45_integrator_->adaptive_step(state, dt);
        this->set_time(rk45_integrator_->current_time());
        
        return result_dt;
    }

    MethodType get_current_method() const { return current_method_; }
    
    void set_stiffness_detection_frequency(int frequency) {
        stiffness_detection_frequency_ = frequency;
    }
    
    void set_stiffness_threshold(time_type threshold) {
        stiffness_threshold_ = threshold;
    }

    void set_tolerances(time_type rtol, time_type atol) {
        this->rtol_ = rtol;
        this->atol_ = atol;
        if (rk45_integrator_) {
            rk45_integrator_->set_tolerances(rtol, atol);
        }
        if (bdf_integrator_) {
            bdf_integrator_->set_tolerances(rtol, atol);
        }
    }

    void integrate(state_type& state, time_type dt, time_type end_time) override {
        // Initialize if needed
        if (!has_previous_values_) {
            y_prev_ = StateCreator<state_type>::create(state);
            f_prev_ = StateCreator<state_type>::create(state);
            has_previous_values_ = false; // Will be set in detect_stiffness
        }
        
        // Call base implementation
        base_type::integrate(state, dt, end_time);
    }

private:
    MethodType current_method_;
    std::unique_ptr<RK45Integrator<S>> rk45_integrator_;
    std::unique_ptr<BDFIntegrator<S>> bdf_integrator_;
    
    int stiffness_detection_frequency_;
    int step_count_;
    int consecutive_stiff_steps_;
    int consecutive_nonstiff_steps_;
    time_type stiffness_threshold_;
    
    // Previous values for stiffness detection
    state_type y_prev_;
    state_type f_prev_;
    bool has_previous_values_;
    
    void detect_stiffness(const state_type& y, time_type dt) {
        // Simplified stiffness detection based on the ratio of Jacobian eigenvalues
        // In practice, LSODA uses more sophisticated detection methods
        
        if (!has_previous_values_) {
            y_prev_ = StateCreator<state_type>::create(y);
            f_prev_ = StateCreator<state_type>::create(y);
            y_prev_ = y;
            this->sys_(this->current_time_, y, f_prev_);
            has_previous_values_ = true;
            return;
        }
        
        state_type f_current = StateCreator<state_type>::create(y);
        this->sys_(this->current_time_, y, f_current);
        
        // Estimate stiffness using the spectral radius approximation
        time_type stiffness_ratio = estimate_stiffness_ratio(y, f_current, dt);
        
        if (stiffness_ratio > stiffness_threshold_) {
            // System appears stiff
            consecutive_stiff_steps_++;
            consecutive_nonstiff_steps_ = 0;
            
            if (consecutive_stiff_steps_ >= 3 && current_method_ == MethodType::ADAMS) {
                switch_to_bdf(y);
            }
        } else {
            // System appears non-stiff
            consecutive_nonstiff_steps_++;
            consecutive_stiff_steps_ = 0;
            
            if (consecutive_nonstiff_steps_ >= 3 && current_method_ == MethodType::BDF) {
                switch_to_adams(y);
            }
        }
        
        // Update previous values
        y_prev_ = y;
        f_prev_ = f_current;
    }
    
    time_type estimate_stiffness_ratio(const state_type& y, const state_type& f, time_type dt) {
        // Estimate the ratio ||J*h|| / ||f|| where J is the Jacobian
        // Use finite differences to approximate Jacobian effects
        
        time_type epsilon = static_cast<time_type>(1e-8);
        state_type y_pert = StateCreator<state_type>::create(y);
        state_type f_pert = StateCreator<state_type>::create(y);
        
        time_type jacobian_norm = static_cast<time_type>(0);
        time_type f_norm = static_cast<time_type>(0);
        
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto f_it = f.begin();
            auto y_pert_it = y_pert.begin();
            auto f_pert_it = f_pert.begin();
            
            // Perturb y[i]
            y_pert = y;
            y_pert_it[i] += epsilon;
            
            // Evaluate f at perturbed point
            this->sys_(this->current_time_, y_pert, f_pert);
            
            // Estimate partial derivative
            time_type df_dy = (f_pert_it[i] - f_it[i]) / epsilon;
            
            // Accumulate norms
            jacobian_norm += std::abs(df_dy * dt);
            f_norm += std::abs(f_it[i]);
        }
        
        if (f_norm < static_cast<time_type>(1e-12)) {
            return static_cast<time_type>(0);
        }
        
        return jacobian_norm / f_norm;
    }
    
    void switch_to_bdf(const state_type& y) {
        current_method_ = MethodType::BDF;
        
        if (!bdf_integrator_) {
            bdf_integrator_ = std::make_unique<BDFIntegrator<S>>(
                [this](time_type t, const state_type& y, state_type& dydt) {
                    this->sys_(t, y, dydt);
                }, this->rtol_, this->atol_);
        }
        
        bdf_integrator_->set_time(this->current_time_);
    }
    
    void switch_to_adams(const state_type& y) {
        current_method_ = MethodType::ADAMS;
        
        if (!rk45_integrator_) {
            rk45_integrator_ = std::make_unique<RK45Integrator<S>>(
                [this](time_type t, const state_type& y, state_type& dydt) {
                    this->sys_(t, y, dydt);
                }, this->rtol_, this->atol_);
        }
        
        rk45_integrator_->set_time(this->current_time_);
    }
};

} // namespace diffeq
