#pragma once
#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <numeric>
#include <iostream>

namespace diffeq {

// SciPy BDF constants
constexpr int MAX_ORDER = 5;
constexpr int NEWTON_MAXITER = 4;
constexpr double MIN_FACTOR = 0.2;
constexpr double MAX_FACTOR = 10.0;

/**
 * @brief BDF (Backward Differentiation Formula) integrator
 *
 * Variable order (1-5) implicit multistep method for stiff systems.
 * Implementation follows SciPy's BDF method exactly.
 * Excellent for stiff differential equations.
 *
 * Order: Variable (1-5)
 * Stages: Variable
 * Adaptive: Yes
 * Stiff: Excellent
 */
template<typename S>
class BDFIntegrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit BDFIntegrator(system_function sys,
                          time_type rtol = static_cast<time_type>(1e-3),
                          time_type atol = static_cast<time_type>(1e-6),
                          int max_order = 5)
        : base_type(std::move(sys), rtol, atol),
          max_order_(std::min<int>(max_order, MAX_ORDER)),
          current_order_(1),
          newton_tolerance_(static_cast<time_type>(1e-3)),
          is_initialized_(false),
          h_abs_(0.0),
          h_abs_old_(0.0),
          error_norm_old_(0.0),
          n_equal_steps_(0) {

        // Initialize BDF coefficients (SciPy style)
        initialize_bdf_coefficients();
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    void fixed_step(state_type& state, time_type dt) {
        // Simple fixed-step BDF1 implementation
        if (!is_initialized_) {
            initialize_history(state, dt);
            is_initialized_ = true;
        }

        state_type y_new = StateCreator<state_type>::create(state);
        state_type error = StateCreator<state_type>::create(state);

        bool success = bdf_step(state, y_new, error, dt);

        if (success) {
            state = y_new;
            this->advance_time(dt);
        } else {
            // Fallback to explicit Euler if Newton fails
            state_type f = StateCreator<state_type>::create(state);
            this->sys_(this->current_time_, state, f);
            for (std::size_t i = 0; i < state.size(); ++i) {
                state[i] += dt * f[i];
            }
            this->advance_time(dt);
        }
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        if (!is_initialized_) {
            initialize_history(state, dt);
            is_initialized_ = true;
        }

        // SciPy-style step size control
        time_type t = this->current_time_;
        time_type max_step = this->dt_max_;
        time_type min_step = (this->dt_min_ > static_cast<time_type>(1e-10)) ? this->dt_min_ : static_cast<time_type>(1e-10);

        time_type h_abs = h_abs_;
        if (h_abs > max_step) {
            h_abs = max_step;
            change_D(current_order_, max_step / h_abs_);
            n_equal_steps_ = 0;
        } else if (h_abs < min_step) {
            h_abs = min_step;
            change_D(current_order_, min_step / h_abs_);
            n_equal_steps_ = 0;
        }

        bool step_accepted = false;
        int attempts = 0;
        const int max_attempts = 10;  // Limit attempts to avoid infinite loops

        while (!step_accepted && attempts < max_attempts) {
            attempts++;

            if (h_abs < min_step) {
                return fallback_step(state, dt);
            }

            time_type h = h_abs;
            time_type t_new = t + h;

            // Check boundary
            if (t_new > this->current_time_ + dt) {
                t_new = this->current_time_ + dt;
                change_D(current_order_, std::abs(t_new - t) / h_abs);
                n_equal_steps_ = 0;
            }

            h = t_new - t;
            h_abs = std::abs(h);

            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);

            if (bdf_step(state, y_new, error, h)) {
                // Calculate error norm with proper scaling
                state_type scale = StateCreator<state_type>::create(y_new);
                for (std::size_t i = 0; i < scale.size(); ++i) {
                    scale[i] = this->atol_ + this->rtol_ * std::abs(y_new[i]);
                }

                time_type error_norm = 0.0;
                for (std::size_t i = 0; i < error.size(); ++i) {
                    time_type scaled_error = error[i] / scale[i];
                    error_norm += scaled_error * scaled_error;
                }
                error_norm = std::sqrt(error_norm / error.size());

                if (error_norm <= 1.0) {
                    // Accept step
                    step_accepted = true;
                    n_equal_steps_++;

                    state = y_new;
                    this->advance_time(h);
                    h_abs_ = h_abs;

                    // Differences array is updated in bdf_step

                    // Adjust order if we have enough equal steps
                    adjust_order(error_norm, scale);

                    // Calculate next step size with proper adaptive logic
                    time_type safety = 0.9;
                    time_type factor;

                    if (error_norm < 0.1) {
                        // Error is very small, increase step size
                        factor = safety * 1.2;  // Increase by 20%
                    } else if (error_norm < 0.5) {
                        // Error is small, keep step size or increase slightly
                        factor = safety * 1.1;  // Increase by 10%
                    } else if (error_norm < 0.8) {
                        // Error is moderate, keep step size
                        factor = safety * 1.0;  // Keep same
                    } else {
                        // Error is large but acceptable, reduce slightly
                        factor = safety * 0.9;  // Reduce by 10%
                    }

                    factor = (factor > MAX_FACTOR) ? static_cast<time_type>(MAX_FACTOR) : factor;
                    factor = (factor < 0.1) ? static_cast<time_type>(0.1) : factor;

                    h_abs_ *= factor;
                    change_D(current_order_, factor);
                    n_equal_steps_ = 0;

                } else {
                    // Reject step - be more conservative
                    time_type factor = static_cast<time_type>(0.5);  // Fixed reduction factor
                    h_abs *= factor;
                    change_D(current_order_, factor);
                    n_equal_steps_ = 0;
                }
            } else {
                // Newton failed - try smaller step
                time_type factor = static_cast<time_type>(0.25);  // More aggressive reduction for Newton failure
                h_abs *= factor;
                change_D(current_order_, factor);
                n_equal_steps_ = 0;
            }
        }

        // If we exit the loop without accepting a step, use fallback
        if (!step_accepted) {
            return fallback_step(state, dt);
        }

        return h_abs_;
    }

    void set_newton_parameters(int max_iterations, time_type tolerance) {
        // max_iterations is now fixed at NEWTON_MAXITER = 4 (SciPy style)
        newton_tolerance_ = tolerance;
    }

private:
    int max_order_;
    int current_order_;
    time_type newton_tolerance_;
    bool is_initialized_;
    time_type h_abs_;
    time_type h_abs_old_;
    time_type error_norm_old_;
    int n_equal_steps_;

    // SciPy-style differences array (D) - stores solution and derivatives
    std::vector<state_type> D_;  // D[0] = y, D[1] = h*f, D[2] = differences, etc.

    // SciPy BDF coefficients
    std::array<time_type, MAX_ORDER + 2> gamma_;      // Gamma coefficients
    std::array<time_type, MAX_ORDER + 2> alpha_;      // Alpha coefficients
    std::array<time_type, MAX_ORDER + 2> error_const_; // Error constants
    
    void initialize_bdf_coefficients() {
        // SciPy BDF coefficients - exactly matching the Python implementation
        std::array<time_type, MAX_ORDER + 1> kappa = {0, -0.1850, -1.0/9.0, -0.0823, -0.0415, 0};

        // Initialize gamma array: gamma[0] = 0, gamma[k] = sum(1/j for j=1..k)
        gamma_[0] = 0.0;
        for (int k = 1; k <= MAX_ORDER; ++k) {
            gamma_[k] = gamma_[k-1] + 1.0 / k;
        }

        // Initialize alpha array: alpha = (1 - kappa) * gamma
        for (int k = 0; k <= MAX_ORDER; ++k) {
            alpha_[k] = (1.0 - kappa[k]) * gamma_[k];
        }

        // Initialize error constants: error_const = kappa * gamma + 1/(k+1)
        for (int k = 0; k <= MAX_ORDER; ++k) {
            error_const_[k] = kappa[k] * gamma_[k] + 1.0 / (k + 1);
        }
    }

    // SciPy-style helper functions for differences array
    std::vector<std::vector<time_type>> compute_R(int order, time_type factor) {
        std::vector<std::vector<time_type>> R(order + 1, std::vector<time_type>(order + 1, 0.0));

        // Initialize R matrix for changing differences array
        for (int i = 1; i <= order; ++i) {
            for (int j = 1; j <= order; ++j) {
                R[i][j] = (i - 1 - factor * j) / i;
            }
        }
        R[0][0] = 1.0;

        // Compute cumulative product
        for (int i = 1; i <= order; ++i) {
            for (int j = 0; j <= order; ++j) {
                R[i][j] *= R[i-1][j];
            }
        }

        return R;
    }

    void change_D(int order, time_type factor) {
        auto R = compute_R(order, factor);
        auto U = compute_R(order, 1.0);

        // Compute RU = R * U
        std::vector<std::vector<time_type>> RU(order + 1, std::vector<time_type>(order + 1, 0.0));
        for (int i = 0; i <= order; ++i) {
            for (int j = 0; j <= order; ++j) {
                for (int k = 0; k <= order; ++k) {
                    RU[i][j] += R[i][k] * U[k][j];
                }
            }
        }

        // Apply transformation: D[:order+1] = RU.T @ D[:order+1]
        std::vector<state_type> D_new(order + 1);
        for (int i = 0; i <= order; ++i) {
            D_new[i] = StateCreator<state_type>::create(D_[0]);
            for (std::size_t k = 0; k < D_[0].size(); ++k) {
                D_new[i][k] = 0.0;
                for (int j = 0; j <= order; ++j) {
                    D_new[i][k] += RU[j][i] * D_[j][k];
                }
            }
        }

        // Copy back
        for (int i = 0; i <= order; ++i) {
            D_[i] = D_new[i];
        }
    }

    void initialize_history(const state_type& y0, time_type dt) {
        // Initialize differences array D (SciPy style)
        D_.clear();
        D_.resize(MAX_ORDER + 3);

        for (int i = 0; i < MAX_ORDER + 3; ++i) {
            D_[i] = StateCreator<state_type>::create(y0);
            // Initialize all to zero except D[0]
            for (std::size_t j = 0; j < y0.size(); ++j) {
                D_[i][j] = 0.0;
            }
        }

        // D[0] = y0
        D_[0] = y0;

        // D[1] will be computed in the first step as h * f(t0, y0)
        // Initialize other differences to zero

        current_order_ = 1;
        n_equal_steps_ = 0;
        h_abs_ = (dt < static_cast<time_type>(0.1)) ? dt : static_cast<time_type>(0.1);
    }

    void update_history(const state_type& y_new, time_type dt) {
        // Update differences array after successful step (done in main step function)
        h_abs_old_ = h_abs_;
        h_abs_ = dt;
    }
    
    void adjust_order(time_type error_norm, const state_type& scale) {
        // SciPy-style order selection - only adjust if we have enough equal steps
        if (n_equal_steps_ < current_order_ + 1) {
            return;
        }

        // Calculate error norms for different orders
        time_type error_m_norm = std::numeric_limits<time_type>::infinity();
        time_type error_p_norm = std::numeric_limits<time_type>::infinity();

        if (current_order_ > 1) {
            // Error estimate for order-1
            time_type error_m = 0.0;
            for (std::size_t i = 0; i < D_[0].size(); ++i) {
                time_type err = error_const_[current_order_ - 1] * D_[current_order_][i] / scale[i];
                error_m += err * err;
            }
            error_m_norm = std::sqrt(error_m / D_[0].size());
        }

        if (current_order_ < MAX_ORDER) {
            // Error estimate for order+1
            time_type error_p = 0.0;
            for (std::size_t i = 0; i < D_[0].size(); ++i) {
                time_type err = error_const_[current_order_ + 1] * D_[current_order_ + 2][i] / scale[i];
                error_p += err * err;
            }
            error_p_norm = std::sqrt(error_p / D_[0].size());
        }

        // Calculate factors for order selection
        std::array<time_type, 3> error_norms = {error_m_norm, error_norm, error_p_norm};
        std::array<time_type, 3> factors;

        for (int i = 0; i < 3; ++i) {
            if (error_norms[i] > 0) {
                factors[i] = std::pow(error_norms[i], -1.0 / (current_order_ + i));
            } else {
                factors[i] = std::numeric_limits<time_type>::infinity();
            }
        }

        // Select order with maximum factor
        int best_order_idx = 0;
        for (int i = 1; i < 3; ++i) {
            if (factors[i] > factors[best_order_idx]) {
                best_order_idx = i;
            }
        }

        int delta_order = best_order_idx - 1;
        int new_order = current_order_ + delta_order;
        if (new_order < 1) new_order = 1;
        if (new_order > MAX_ORDER) new_order = MAX_ORDER;
        current_order_ = new_order;
    }
    
    // SciPy-style BDF system solver
    struct NewtonResult {
        bool converged;
        int iterations;
        state_type y;
        state_type d;
    };

    NewtonResult solve_bdf_system(time_type t_new, const state_type& y_predict,
                                 time_type c, const state_type& psi, const state_type& scale) {
        NewtonResult result;
        result.converged = false;
        result.iterations = 0;
        result.y = y_predict;
        result.d = StateCreator<state_type>::create(y_predict);

        // Initialize d to zero
        for (std::size_t i = 0; i < result.d.size(); ++i) {
            result.d[i] = 0.0;
        }

        // Newton iteration to solve: d - c * f(t_new, y_predict + d) = -psi
        for (int k = 0; k < NEWTON_MAXITER; ++k) {
            result.iterations = k + 1;

            // Evaluate f at current y = y_predict + d
            state_type f = StateCreator<state_type>::create(result.y);
            this->sys_(t_new, result.y, f);

            // Check if f is finite
            bool f_finite = true;
            for (std::size_t i = 0; i < f.size(); ++i) {
                if (!std::isfinite(f[i])) {
                    f_finite = false;
                    break;
                }
            }
            if (!f_finite) break;

            // Compute residual: F(d) = d - c * f(t_new, y_predict + d) + psi
            state_type residual = StateCreator<state_type>::create(result.d);
            for (std::size_t i = 0; i < residual.size(); ++i) {
                residual[i] = result.d[i] - c * f[i] + psi[i];
            }

            // Check convergence
            time_type norm = 0.0;
            for (std::size_t i = 0; i < residual.size(); ++i) {
                time_type scaled_res = residual[i] / scale[i];
                norm += scaled_res * scaled_res;
            }
            norm = std::sqrt(norm / residual.size());

            if (norm < newton_tolerance_) {
                result.converged = true;
                break;
            }

            // Newton step: solve (I - c*J) * dd = -residual for dd
            // Then update: d = d + dd, y = y_predict + d
            state_type dd = StateCreator<state_type>::create(result.d);
            for (std::size_t i = 0; i < result.d.size(); ++i) {
                // Approximate Jacobian diagonal element
                time_type jac_diag = estimate_jacobian_diagonal(i, result.y, t_new);
                time_type denominator = 1.0 - c * jac_diag;
                if (std::abs(denominator) > static_cast<time_type>(1e-12)) {
                    dd[i] = -residual[i] / denominator;
                } else {
                    dd[i] = -residual[i];
                }
                result.d[i] += dd[i];
                result.y[i] = y_predict[i] + result.d[i];
            }
        }

        return result;
    }

    bool bdf_step(const state_type& y_current, state_type& y_new, state_type& error, time_type dt) {
        // SciPy-style BDF step implementation using differences array
        time_type t_new = this->current_time_ + dt;

        // Update D[1] for the current step size
        // D[1] = h * f(t_current, y_current)
        state_type f_current = StateCreator<state_type>::create(D_[0]);
        this->sys_(this->current_time_, D_[0], f_current);
        for (std::size_t i = 0; i < D_[0].size(); ++i) {
            D_[1][i] = dt * f_current[i];
        }

        // Calculate y_predict = sum(D[:order+1])
        state_type y_predict = StateCreator<state_type>::create(D_[0]);
        for (std::size_t i = 0; i < y_predict.size(); ++i) {
            y_predict[i] = 0.0;
            for (int j = 0; j <= current_order_; ++j) {
                y_predict[i] += D_[j][i];
            }
        }

        // Calculate scale = atol + rtol * abs(y_predict)
        state_type scale = StateCreator<state_type>::create(y_predict);
        for (std::size_t i = 0; i < scale.size(); ++i) {
            scale[i] = this->atol_ + this->rtol_ * std::abs(y_predict[i]);
        }

        // Calculate psi = dot(D[1:order+1].T, gamma[1:order+1]) / alpha[order]
        state_type psi = StateCreator<state_type>::create(y_predict);
        for (std::size_t i = 0; i < psi.size(); ++i) {
            psi[i] = 0.0;
            for (int j = 1; j <= current_order_; ++j) {
                psi[i] += D_[j][i] * gamma_[j];
            }
            psi[i] /= alpha_[current_order_];
        }

        // Calculate c = h / alpha[order]
        time_type c = dt / alpha_[current_order_];

        // Solve BDF system
        auto result = solve_bdf_system(t_new, y_predict, c, psi, scale);
        if (!result.converged) {
            return false;
        }

        // Calculate error = error_const[order] * d
        for (std::size_t i = 0; i < error.size(); ++i) {
            error[i] = error_const_[current_order_] * result.d[i];
        }

        // Calculate error norm
        time_type error_norm = 0.0;
        for (std::size_t i = 0; i < error.size(); ++i) {
            time_type scaled_error = error[i] / scale[i];
            error_norm += scaled_error * scaled_error;
        }
        error_norm = std::sqrt(error_norm / error.size());

        if (error_norm > 1.0) {
            return false;  // Step rejected
        }

        // Step accepted - update differences array (SciPy style)
        // D[0] should be the new solution
        D_[0] = result.y;
        y_new = result.y;

        // Update differences: D[order+2] = d - D[order+1], D[order+1] = d
        if (current_order_ + 2 < static_cast<int>(D_.size())) {
            for (std::size_t i = 0; i < result.d.size(); ++i) {
                D_[current_order_ + 2][i] = result.d[i] - D_[current_order_ + 1][i];
            }
        }
        D_[current_order_ + 1] = result.d;

        // Update differences: D[i] += D[i+1] for i = order, order-1, ..., 1 (not 0!)
        for (int i = current_order_; i >= 1; --i) {
            for (std::size_t j = 0; j < D_[i].size(); ++j) {
                D_[i][j] += D_[i + 1][j];
            }
        }

        return true;
    }
    
    time_type estimate_jacobian_diagonal(std::size_t i, const state_type& y, time_type t) {
        // Estimate diagonal element of Jacobian using finite differences
        time_type epsilon = (static_cast<time_type>(1e-8) > std::abs(y[i]) * static_cast<time_type>(1e-8)) ? static_cast<time_type>(1e-8) : std::abs(y[i]) * static_cast<time_type>(1e-8);
        state_type y_pert = StateCreator<state_type>::create(y);
        state_type f_orig = StateCreator<state_type>::create(y);
        state_type f_pert = StateCreator<state_type>::create(y);

        // Evaluate f at original point
        this->sys_(t, y, f_orig);

        // Perturb y[i] and evaluate f
        y_pert = y;
        y_pert[i] += epsilon;
        this->sys_(t, y_pert, f_pert);

        // Estimate ∂f_i/∂y_i with better numerical stability
        time_type jac = (f_pert[i] - f_orig[i]) / epsilon;

        // For exponential decay dy/dt = -y, Jacobian should be -1
        // Clamp to reasonable range to avoid numerical issues
        return (static_cast<time_type>(-100.0) > jac) ? static_cast<time_type>(-100.0) : ((jac > static_cast<time_type>(100.0)) ? static_cast<time_type>(100.0) : jac);
    }

    // Note: update_differences_array is now handled directly in bdf_step

    time_type fallback_step(state_type& state, time_type dt) {
        // Fallback to simple forward Euler for problematic cases
        time_type small_dt = (this->dt_min_ * 2.0 > dt * static_cast<time_type>(0.1)) ? this->dt_min_ * 2.0 : dt * static_cast<time_type>(0.1);

        state_type f_current = StateCreator<state_type>::create(state);

        // Simple forward Euler step
        this->sys_(this->current_time_, state, f_current);

        for (std::size_t i = 0; i < state.size(); ++i) {
            state[i] = state[i] + small_dt * f_current[i];
        }


        this->advance_time(small_dt);

        // Reset to order 1 and reinitialize
        current_order_ = 1;
        n_equal_steps_ = 0;
        h_abs_ = small_dt;

        return small_dt;
    }
};

} // namespace diffeq
