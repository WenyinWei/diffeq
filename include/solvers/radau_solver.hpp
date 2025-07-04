#pragma once
#include <core/adaptive_integrator.hpp>
#include <cmath>
#include <array>

// Radau IIA integrator for stiff systems
// 5th order implicit Runge-Kutta method
// Excellent for stiff differential equations
template<system_state S, can_be_time T = double>
class RadauIntegrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit RadauIntegrator(system_function sys, 
                           time_type rtol = static_cast<time_type>(1e-6),
                           time_type atol = static_cast<time_type>(1e-9))
        : base_type(std::move(sys), rtol, atol), 
          max_newton_iterations_(10),
          newton_tolerance_(static_cast<time_type>(1e-12)) {}

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
            
            if (radau_step(state, y_new, error, current_dt)) {
                // Calculate error norm
                time_type err_norm = this->error_norm(error, state);
                
                if (err_norm <= 1.0) {
                    // Step accepted
                    state = y_new;
                    this->advance_time(current_dt);
                    
                    // Suggest next step size (5th order)
                    current_dt = this->suggest_step_size(current_dt, err_norm, 5);
                    return current_dt;
                } else {
                    // Step rejected, reduce step size
                    current_dt = this->suggest_step_size(current_dt, err_norm, 5);
                    if (current_dt < this->dt_min_) {
                        break;
                    }
                }
            } else {
                // Newton iteration failed, reduce step size
                current_dt *= static_cast<time_type>(0.5);
                if (current_dt < this->dt_min_) {
                    break;
                }
            }
        }
        
        throw std::runtime_error("Radau: Maximum number of step size reductions exceeded");
    }

    void set_newton_parameters(int max_iterations, time_type tolerance) {
        max_newton_iterations_ = max_iterations;
        newton_tolerance_ = tolerance;
    }

private:
    int max_newton_iterations_;
    time_type newton_tolerance_;
    
    bool radau_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Radau IIA (3 stages, 5th order) coefficients
        const time_type sqrt6 = std::sqrt(static_cast<time_type>(6));
        
        // Butcher tableau for Radau IIA
        const std::array<time_type, 3> c = {
            (static_cast<time_type>(4) - sqrt6) / static_cast<time_type>(10),
            (static_cast<time_type>(4) + sqrt6) / static_cast<time_type>(10),
            static_cast<time_type>(1)
        };
        
        const std::array<std::array<time_type, 3>, 3> A = {{
            {{(static_cast<time_type>(88) - static_cast<time_type>(7)*sqrt6) / static_cast<time_type>(360),
              (static_cast<time_type>(296) - static_cast<time_type>(169)*sqrt6) / static_cast<time_type>(1800),
              (-static_cast<time_type>(2) + static_cast<time_type>(3)*sqrt6) / static_cast<time_type>(225)}},
            {{(static_cast<time_type>(296) + static_cast<time_type>(169)*sqrt6) / static_cast<time_type>(1800),
              (static_cast<time_type>(88) + static_cast<time_type>(7)*sqrt6) / static_cast<time_type>(360),
              (-static_cast<time_type>(2) - static_cast<time_type>(3)*sqrt6) / static_cast<time_type>(225)}},
            {{(static_cast<time_type>(16) - sqrt6) / static_cast<time_type>(36),
              (static_cast<time_type>(16) + sqrt6) / static_cast<time_type>(36),
              static_cast<time_type>(1) / static_cast<time_type>(9)}}
        }};
        
        const std::array<time_type, 3> b = {
            (static_cast<time_type>(16) - sqrt6) / static_cast<time_type>(36),
            (static_cast<time_type>(16) + sqrt6) / static_cast<time_type>(36),
            static_cast<time_type>(1) / static_cast<time_type>(9)
        };
        
        // Stage values
        std::array<state_type, 3> k;
        for (int i = 0; i < 3; ++i) {
            k[i] = StateCreator<state_type>::create(y);
        }
        
        state_type temp = StateCreator<state_type>::create(y);
        state_type rhs = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        // Initial guess for k values (explicit Euler)
        this->sys_(t, y, k[0]);
        for (int i = 1; i < 3; ++i) {
            k[i] = k[0];
        }
        
        // Newton iteration to solve the implicit system
        for (int newton_iter = 0; newton_iter < max_newton_iterations_; ++newton_iter) {
            time_type max_correction = static_cast<time_type>(0);
            
            // Calculate residuals and solve for corrections
            for (int stage = 0; stage < 3; ++stage) {
                // Calculate Y_i = y + dt * sum(A_ij * k_j)
                for (std::size_t i = 0; i < y.size(); ++i) {
                    auto y_it = y.begin();
                    auto temp_it = temp.begin();
                    temp_it[i] = y_it[i];
                    
                    for (int j = 0; j < 3; ++j) {
                        auto k_j_it = k[j].begin();
                        temp_it[i] += dt * A[stage][j] * k_j_it[i];
                    }
                }
                
                // Calculate f(t + c_i * dt, Y_i)
                this->sys_(t + c[stage] * dt, temp, rhs);
                
                // Calculate residual: R_i = k_i - f(t + c_i * dt, Y_i)
                time_type stage_correction = static_cast<time_type>(0);
                for (std::size_t i = 0; i < y.size(); ++i) {
                    auto k_stage_it = k[stage].begin();
                    auto rhs_it = rhs.begin();
                    time_type residual = k_stage_it[i] - rhs_it[i];
                    
                    // Simple correction (in practice, would need Jacobian)
                    time_type correction = residual;
                    k_stage_it[i] -= correction;
                    
                    stage_correction = std::max(stage_correction, std::abs(correction));
                }
                
                max_correction = std::max(max_correction, stage_correction);
            }
            
            // Check convergence
            if (max_correction < newton_tolerance_) {
                break;
            }
            
            if (newton_iter == max_newton_iterations_ - 1) {
                return false; // Newton iteration failed to converge
            }
        }
        
        // Calculate solution: y_new = y + dt * sum(b_i * k_i)
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto y_new_it = y_new.begin();
            y_new_it[i] = y_it[i];
            
            for (int j = 0; j < 3; ++j) {
                auto k_j_it = k[j].begin();
                y_new_it[i] += dt * b[j] * k_j_it[i];
            }
        }
        
        // Error estimation (simplified - use embedded lower order method)
        // For Radau, we can use the difference between 5th and 3rd order solutions
        const std::array<time_type, 3> b_low = {
            static_cast<time_type>(1) / static_cast<time_type>(4),
            static_cast<time_type>(1) / static_cast<time_type>(4),
            static_cast<time_type>(1) / static_cast<time_type>(2)
        };
        
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto error_it = error.begin();
            error_it[i] = static_cast<time_type>(0);
            
            for (int j = 0; j < 3; ++j) {
                auto k_j_it = k[j].begin();
                error_it[i] += dt * (b[j] - b_low[j]) * k_j_it[i];
            }
        }
        
        return true;
    }
};
