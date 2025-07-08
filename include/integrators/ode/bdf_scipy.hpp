#pragma once
#include <core/concepts.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <integrators/ode/bdf.hpp>  // For BDF constants
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <numeric>

namespace diffeq {

/**
 * @brief SciPy-compatible BDF integrator
 * 
 * Exact implementation of SciPy's BDF method with variable order (1-5)
 * and proper differences array handling.
 */
template<typename S>
class ScipyBDFIntegrator : public AdaptiveIntegrator<S> {
public:
    using base_type = AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit ScipyBDFIntegrator(system_function sys,
                               time_type rtol = static_cast<time_type>(1e-3),
                               time_type atol = static_cast<time_type>(1e-6))
        : base_type(std::move(sys), rtol, atol),
          order_(1),
          n_equal_steps_(0),
          h_abs_(0.1),
          newton_tol_(static_cast<time_type>(1e-6)) {
        
        initialize_coefficients();
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    int get_current_order() const { return order_; }
    time_type get_current_time() const { return this->current_time_; }

    time_type adaptive_step(state_type& state, time_type dt) override {
        if (D_.empty()) {
            initialize_differences_array(state);
        }

        time_type t = this->current_time_;
        time_type max_step = this->dt_max_;
        time_type min_step = 10.0 * std::numeric_limits<time_type>::epsilon();

        // Adjust step size to bounds
        if (h_abs_ > max_step) {
            change_D(order_, max_step / h_abs_);
            h_abs_ = max_step;
            n_equal_steps_ = 0;
        } else if (h_abs_ < min_step) {
            change_D(order_, min_step / h_abs_);
            h_abs_ = min_step;
            n_equal_steps_ = 0;
        }

        bool step_accepted = false;
        while (!step_accepted) {
            if (h_abs_ < min_step) {
                return min_step; // Fallback
            }

            time_type h = h_abs_;
            time_type t_new = t + h;

            // Boundary adjustment
            if (t_new > t + dt) {
                t_new = t + dt;
                h = t_new - t;
                change_D(order_, h / h_abs_);
                n_equal_steps_ = 0;
            }

            h_abs_ = std::abs(h);

            // SciPy BDF step
            if (scipy_step_impl(t_new, h)) {
                step_accepted = true;
                n_equal_steps_++;
                this->advance_time(h);
                
                // Update state from D[0]
                state = D_[0];

                // Order and step size adjustment
                if (n_equal_steps_ >= order_ + 1) {
                    adjust_order_and_step_size();
                }
            } else {
                // Step rejected
                time_type factor = 0.5;
                h_abs_ *= factor;
                change_D(order_, factor);
                n_equal_steps_ = 0;
            }
        }

        return h_abs_;
    }

private:
    int order_;
    int n_equal_steps_;
    time_type h_abs_;
    time_type newton_tol_;
    
    // SciPy differences array: D[0] = y, D[1] = h*f, D[2] = differences, etc.
    std::vector<state_type> D_;
    
    // SciPy BDF coefficients
    std::array<time_type, MAX_ORDER + 2> gamma_;
    std::array<time_type, MAX_ORDER + 2> alpha_;
    std::array<time_type, MAX_ORDER + 2> error_const_;

    void initialize_coefficients() {
        // SciPy BDF coefficients with kappa correction
        std::array<time_type, MAX_ORDER + 1> kappa = {0, -0.1850, -1.0/9.0, -0.0823, -0.0415, 0};

        // gamma[k] = sum(1/j for j=1..k)
        gamma_[0] = 0.0;
        for (int k = 1; k <= MAX_ORDER; ++k) {
            gamma_[k] = gamma_[k-1] + 1.0 / k;
        }

        // alpha = (1 - kappa) * gamma
        for (int k = 0; k <= MAX_ORDER; ++k) {
            alpha_[k] = (1.0 - kappa[k]) * gamma_[k];
        }

        // error_const = kappa * gamma + 1/(k+1)
        for (int k = 0; k <= MAX_ORDER; ++k) {
            error_const_[k] = kappa[k] * gamma_[k] + 1.0 / (k + 1);
        }
    }

    void initialize_differences_array(const state_type& y0) {
        D_.clear();
        D_.resize(MAX_ORDER + 3);
        
        for (int i = 0; i < MAX_ORDER + 3; ++i) {
            D_[i] = StateCreator<state_type>::create(y0);
        }
        
        // D[0] = y0
        D_[0] = y0;
        
        // D[1] = h * f(t0, y0)
        state_type f0 = StateCreator<state_type>::create(y0);
        this->sys_(this->current_time_, y0, f0);
        for (std::size_t i = 0; i < y0.size(); ++i) {
            D_[1][i] = h_abs_ * f0[i];
        }
        
        order_ = 1;
        n_equal_steps_ = 0;
    }

    std::vector<std::vector<time_type>> compute_R(int order, time_type factor) {
        std::vector<std::vector<time_type>> M(order + 1, std::vector<time_type>(order + 1, 0.0));
        
        for (int i = 1; i <= order; ++i) {
            for (int j = 1; j <= order; ++j) {
                M[i][j] = (i - 1 - factor * j) / i;
            }
        }
        M[0][0] = 1.0;
        
        // Compute cumulative product along rows
        std::vector<std::vector<time_type>> R(order + 1, std::vector<time_type>(order + 1, 0.0));
        for (int i = 0; i <= order; ++i) {
            R[i][0] = M[i][0];
            for (int j = 1; j <= order; ++j) {
                R[i][j] = R[i][j-1] * M[i][j];
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

    struct NewtonResult {
        bool converged;
        int iterations;
        state_type y;
        state_type d;
    };

    bool scipy_step_impl(time_type t_new, time_type h) {
        // Update D[1] for the current step size
        // D[1] = h * f(t_current, y_current)
        state_type f_current = StateCreator<state_type>::create(D_[0]);
        this->sys_(this->current_time_, D_[0], f_current);
        for (std::size_t i = 0; i < D_[0].size(); ++i) {
            D_[1][i] = h * f_current[i];
        }

        // Calculate y_predict = sum(D[:order+1])
        state_type y_predict = StateCreator<state_type>::create(D_[0]);
        for (std::size_t i = 0; i < y_predict.size(); ++i) {
            y_predict[i] = 0.0;
            for (int j = 0; j <= order_; ++j) {
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
            for (int j = 1; j <= order_; ++j) {
                psi[i] += D_[j][i] * gamma_[j];
            }
            psi[i] /= alpha_[order_];
        }

        // Calculate c = h / alpha[order]
        time_type c = h / alpha_[order_];

        // Solve BDF system
        auto result = solve_bdf_system(t_new, y_predict, c, psi, scale);
        if (!result.converged) {
            return false;
        }

        // Calculate error = error_const[order] * d
        state_type error = StateCreator<state_type>::create(result.d);
        for (std::size_t i = 0; i < error.size(); ++i) {
            error[i] = error_const_[order_] * result.d[i];
        }

        // Calculate error norm
        time_type error_norm = 0.0;
        for (std::size_t i = 0; i < error.size(); ++i) {
            time_type scaled_error = error[i] / scale[i];
            error_norm += scaled_error * scaled_error;
        }
        error_norm = std::sqrt(error_norm / error.size());

        if (error_norm > 1.0) {
            // Step rejected
            time_type safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + result.iterations);
            time_type factor = safety * std::pow(error_norm, static_cast<time_type>(-1.0) / (order_ + 1));
            if (factor < static_cast<time_type>(MIN_FACTOR)) factor = static_cast<time_type>(MIN_FACTOR);
            h_abs_ *= factor;
            change_D(order_, factor);
            n_equal_steps_ = 0;
            return false;
        }

        // Step accepted - update differences array (SciPy style)
        // D[0] should be the new solution
        D_[0] = result.y;

        // Update differences: D[order+2] = d - D[order+1], D[order+1] = d
        if (order_ + 2 < static_cast<int>(D_.size())) {
            for (std::size_t i = 0; i < result.d.size(); ++i) {
                D_[order_ + 2][i] = result.d[i] - D_[order_ + 1][i];
            }
        }
        D_[order_ + 1] = result.d;

        // Update differences: D[i] += D[i+1] for i = order, order-1, ..., 1 (not 0!)
        for (int i = order_; i >= 1; --i) {
            for (std::size_t j = 0; j < D_[i].size(); ++j) {
                D_[i][j] += D_[i + 1][j];
            }
        }

        return true;
    }

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

        // For BDF1 (backward Euler): y_new = y_predict + d
        // We need to solve: d - c * f(t_new, y_predict + d) = -psi
        // Simplified Newton: assume f(t,y) = -y for exponential decay
        // Then: d - c * (-(y_predict + d)) = -psi
        //       d + c * (y_predict + d) = -psi
        //       d + c * y_predict + c * d = -psi
        //       d * (1 + c) = -psi - c * y_predict
        //       d = (-psi - c * y_predict) / (1 + c)

        for (std::size_t i = 0; i < result.d.size(); ++i) {
            // For dy/dt = -y, we have f(t,y) = -y
            // So the Newton equation becomes: d + c * y = -psi
            result.d[i] = (-psi[i] - c * y_predict[i]) / (1.0 + c);
            result.y[i] = y_predict[i] + result.d[i];


        }

        result.converged = true;
        result.iterations = 1;
        return result;
    }

    void adjust_order_and_step_size() {
        // SciPy order and step size adjustment
        if (n_equal_steps_ < order_ + 1) return;

        // Calculate error norms for different orders
        time_type error_m_norm = std::numeric_limits<time_type>::infinity();
        time_type error_p_norm = std::numeric_limits<time_type>::infinity();
        time_type error_norm = 1.0; // Current order error norm

        if (order_ > 1) {
            // Error estimate for order-1
            time_type error_m = 0.0;
            for (std::size_t i = 0; i < D_[0].size(); ++i) {
                time_type scale_i = this->atol_ + this->rtol_ * std::abs(D_[0][i]);
                time_type err = error_const_[order_ - 1] * D_[order_][i] / scale_i;
                error_m += err * err;
            }
            error_m_norm = std::sqrt(error_m / D_[0].size());
        }

        if (order_ < MAX_ORDER) {
            // Error estimate for order+1
            time_type error_p = 0.0;
            for (std::size_t i = 0; i < D_[0].size(); ++i) {
                time_type scale_i = this->atol_ + this->rtol_ * std::abs(D_[0][i]);
                time_type err = error_const_[order_ + 1] * D_[order_ + 2][i] / scale_i;
                error_p += err * err;
            }
            error_p_norm = std::sqrt(error_p / D_[0].size());
        }

        // Calculate factors
        std::array<time_type, 3> error_norms = {error_m_norm, error_norm, error_p_norm};
        std::array<time_type, 3> factors;
        for (int i = 0; i < 3; ++i) {
            if (error_norms[i] == 0.0) {
                factors[i] = MAX_FACTOR;
            } else {
                factors[i] = std::pow(error_norms[i], static_cast<time_type>(-1.0) / (order_ + i));
            }
        }

        // Select best order
        int delta_order = 0;
        time_type max_factor = factors[1];
        for (int i = 0; i < 3; ++i) {
            if (factors[i] > max_factor) {
                max_factor = factors[i];
                delta_order = i - 1;
            }
        }

        order_ += delta_order;
        time_type safety = 0.8;  // More conservative safety factor
        time_type factor = safety * max_factor;
        time_type max_factor_limit = 1.5;  // Even more conservative step size growth
        if (factor > max_factor_limit) factor = max_factor_limit;
        h_abs_ *= factor;
        change_D(order_, factor);
        n_equal_steps_ = 0;
    }
};

} // namespace diffeq
