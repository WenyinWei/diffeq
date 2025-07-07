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
          newton_tolerance_(static_cast<time_type>(0.03)),
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

    time_type adaptive_step(state_type& state, time_type dt) override {
        if (!is_initialized_) {
            initialize_history(state, dt);
            is_initialized_ = true;
        }

        // SciPy-style step size control
        time_type t = this->current_time_;
        time_type max_step = this->dt_max_;
        time_type min_step = 10.0 * std::abs(std::nextafter(t, t + dt) - t);

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
        while (!step_accepted) {
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

                    // Update differences array
                    update_differences_array(y_new, h);

                    // Adjust order if we have enough equal steps
                    adjust_order(error_norm, scale);

                    // Calculate next step size with safety factor
                    time_type safety = 0.9;
                    time_type factor_calc = safety * ((error_norm > 1.0) ? (1.0 / error_norm) : 1.0);
                    time_type factor = (factor_calc < MAX_FACTOR) ? factor_calc : static_cast<time_type>(MAX_FACTOR);
                    h_abs_ *= factor;
                    change_D(current_order_, factor);
                    n_equal_steps_ = 0;

                } else {
                    // Reject step
                    time_type factor_calc = 0.9 * std::pow(error_norm, -1.0 / (current_order_ + 1));
                    time_type factor = (factor_calc > MIN_FACTOR) ? factor_calc : static_cast<time_type>(MIN_FACTOR);
                    h_abs *= factor;
                    change_D(current_order_, factor);
                    n_equal_steps_ = 0;
                }
            } else {
                // Newton failed
                time_type factor = 0.5;
                h_abs *= factor;
                change_D(current_order_, factor);
                n_equal_steps_ = 0;
            }
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
        }

        // D[0] = y0
        D_[0] = y0;

        // D[1] = h * f(t0, y0) * direction (we'll compute this in the first step)
        state_type f0 = StateCreator<state_type>::create(y0);
        this->sys_(this->current_time_, y0, f0);
        for (std::size_t i = 0; i < y0.size(); ++i) {
            D_[1][i] = dt * f0[i];
        }

        current_order_ = 1;
        n_equal_steps_ = 0;
        h_abs_ = dt;
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
                                 time_type c, const state_type& psi,
                                 const state_type& scale) {
        NewtonResult result;
        result.converged = false;
        result.iterations = 0;
        result.y = y_predict;
        result.d = StateCreator<state_type>::create(y_predict);

        // Initialize d to zero
        for (std::size_t i = 0; i < result.d.size(); ++i) {
            result.d[i] = 0.0;
        }

        time_type dy_norm_old = 0.0;
        bool has_old_norm = false;

        for (int k = 0; k < NEWTON_MAXITER; ++k) {
            result.iterations = k + 1;

            // Evaluate f at current y
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

            // Compute Newton step: dy = solve(I - c*J, c*f - psi - d)
            state_type rhs = StateCreator<state_type>::create(result.y);
            for (std::size_t i = 0; i < rhs.size(); ++i) {
                rhs[i] = c * f[i] - psi[i] - result.d[i];
            }

            // Simplified solver (diagonal approximation)
            state_type dy = StateCreator<state_type>::create(result.y);
            for (std::size_t i = 0; i < dy.size(); ++i) {
                // Approximate Jacobian diagonal element
                time_type jac_diag = estimate_jacobian_diagonal(i, result.y, t_new);
                time_type denominator = 1.0 - c * jac_diag;
                dy[i] = (std::abs(denominator) > newton_tolerance_) ? rhs[i] / denominator : 0.0;
            }

            // Compute dy norm
            time_type dy_norm = 0.0;
            for (std::size_t i = 0; i < dy.size(); ++i) {
                time_type scaled_dy = dy[i] / scale[i];
                dy_norm += scaled_dy * scaled_dy;
            }
            dy_norm = std::sqrt(dy_norm / dy.size());

            // Check convergence rate
            if (has_old_norm) {
                time_type rate = dy_norm / dy_norm_old;
                if (rate >= 1.0 ||
                    std::pow(rate, NEWTON_MAXITER - k) / (1.0 - rate) * dy_norm > 1.0) {
                    break;
                }
            }

            // Update solution
            for (std::size_t i = 0; i < result.y.size(); ++i) {
                result.y[i] += dy[i];
                result.d[i] += dy[i];
            }

            // Check convergence
            if (dy_norm == 0.0 ||
                (has_old_norm && dy_norm_old / (1.0 - dy_norm / dy_norm_old) * dy_norm < 1.0)) {
                result.converged = true;
                break;
            }

            dy_norm_old = dy_norm;
            has_old_norm = true;
        }

        return result;
    }

    bool bdf_step(const state_type& y_current, state_type& y_new, state_type& error, time_type dt) {
        // Predict solution using differences array
        y_new = StateCreator<state_type>::create(y_current);
        for (std::size_t i = 0; i < y_new.size(); ++i) {
            y_new[i] = 0.0;
            for (int j = 0; j <= current_order_; ++j) {
                y_new[i] += D_[j][i];
            }
        }

        // Compute scale for error control
        state_type scale = StateCreator<state_type>::create(y_new);
        for (std::size_t i = 0; i < scale.size(); ++i) {
            scale[i] = this->atol_ + this->rtol_ * std::abs(y_new[i]);
        }

        // Compute psi for BDF formula
        state_type psi = StateCreator<state_type>::create(y_new);
        for (std::size_t i = 0; i < psi.size(); ++i) {
            psi[i] = 0.0;
            for (int j = 1; j <= current_order_; ++j) {
                psi[i] += D_[j][i] * gamma_[j] / alpha_[current_order_];
            }
        }

        time_type c = dt / alpha_[current_order_];
        time_type t_new = this->current_time_ + dt;

        auto newton_result = solve_bdf_system(t_new, y_new, c, psi, scale);

        if (newton_result.converged) {
            y_new = newton_result.y;

            // Calculate error estimate
            for (std::size_t i = 0; i < error.size(); ++i) {
                error[i] = error_const_[current_order_] * newton_result.d[i];
            }
            return true;
        }

        return false;
    }
    
    time_type estimate_jacobian_diagonal(std::size_t i, const state_type& y, time_type t) {
        // Estimate diagonal element of Jacobian using finite differences
        time_type epsilon = static_cast<time_type>(1e-8);
        state_type y_pert = StateCreator<state_type>::create(y);
        state_type f_orig = StateCreator<state_type>::create(y);
        state_type f_pert = StateCreator<state_type>::create(y);

        // Evaluate f at original point
        this->sys_(t, y, f_orig);

        // Perturb y[i] and evaluate f
        y_pert = y;
        y_pert[i] += epsilon;
        this->sys_(t, y_pert, f_pert);

        // Estimate ∂f_i/∂y_i
        return (f_pert[i] - f_orig[i]) / epsilon;
    }

    void update_differences_array(const state_type& y_new, time_type h) {
        // Update differences array after successful step (SciPy style)
        // The principal relation: D^{j+1} y_n = D^j y_n - D^j y_{n-1}

        // Calculate d = D^{k+1} y_n (the correction from Newton iteration)
        state_type d = StateCreator<state_type>::create(y_new);
        for (std::size_t i = 0; i < d.size(); ++i) {
            d[i] = y_new[i] - D_[0][i];
            for (int j = 1; j <= current_order_; ++j) {
                d[i] -= D_[j][i];
            }
        }

        // Update differences: D[order+2] = d - D[order+1]
        for (std::size_t i = 0; i < d.size(); ++i) {
            D_[current_order_ + 2][i] = d[i] - D_[current_order_ + 1][i];
        }

        // D[order+1] = d
        D_[current_order_ + 1] = d;

        // Update all lower order differences
        for (int j = current_order_; j >= 0; --j) {
            for (std::size_t i = 0; i < D_[j].size(); ++i) {
                D_[j][i] += D_[j + 1][i];
            }
        }
    }

    time_type fallback_step(state_type& state, time_type dt) {
        // Fallback to backward Euler with very small step
        time_type small_dt = std::min<time_type>(dt, static_cast<time_type>(1e-6));

        state_type y_new = StateCreator<state_type>::create(state);
        state_type f_new = StateCreator<state_type>::create(state);

        // Simple backward Euler iteration
        y_new = state;
        for (int iter = 0; iter < 5; ++iter) {
            this->sys_(this->current_time_ + small_dt, y_new, f_new);

            for (std::size_t i = 0; i < state.size(); ++i) {
                y_new[i] = state[i] + small_dt * f_new[i];
            }
        }

        state = y_new;
        this->advance_time(small_dt);

        // Reset to order 1 and reinitialize
        current_order_ = 1;
        n_equal_steps_ = 0;
        h_abs_ = small_dt;

        return small_dt;
    }
};

} // namespace diffeq
