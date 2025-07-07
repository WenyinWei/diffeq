#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <integrators/ode/dop853_coefficients.hpp>
#include <cmath>
#include <stdexcept>

namespace diffeq {

template<system_state S>
class DOP853Integrator;

template<system_state S>
class DOP853DenseOutputHelper {
public:
    using value_type = typename DOP853Integrator<S>::value_type;
    
    // Dense output for DOP853: ported from Fortran CONTD8
    // CON: continuous output coefficients, size 8*nd
    // ICOMP: index mapping, size nd
    // nd: number of dense output components
    // xold: left endpoint of interval, h: step size
    // x: interpolation point (xold <= x <= xold+h)
    // Returns: interpolated value for component ii at x
    static value_type contd8(
        int ii, value_type x, const value_type* con, const int* icomp, int nd,
        value_type xold, value_type h) {
        int i = -1;
        for (int j = 0; j < nd; ++j) {
            if (icomp[j] == ii) { i = j; break; }
        }
        if (i == -1) {
            throw std::runtime_error("No dense output available for component " + std::to_string(ii));
        }
        value_type s = (x - xold) / h;
        value_type s1 = 1.0 - s;
        value_type conpar = con[i + nd*4] + s * (con[i + nd*5] + s1 * (con[i + nd*6] + s * con[i + nd*7]));
        value_type result = con[i] + s * (con[i + nd] + s1 * (con[i + nd*2] + s * (con[i + nd*3] + s1 * conpar)));
        return result;
    }
};


/**
 * @brief DOP853 (Dormand-Prince 8(5,3)) adaptive integrator
 * 
 * Eighth-order method with embedded 5th and 3rd order error estimation.
 * Reference: Hairer, Norsett, Wanner, "Solving Ordinary Differential Equations I"
 */
template<system_state S>
class DOP853Integrator : public AdaptiveIntegrator<S> {
    

public:
    using base_type = AdaptiveIntegrator<S>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    // Fortran default parameters (do not change unless you know what you are doing)
    static constexpr time_type fortran_safety = static_cast<time_type>(0.9);   // SAFE
    static constexpr time_type fortran_fac1 = static_cast<time_type>(0.333);   // FAC1 (min step size factor)
    static constexpr time_type fortran_fac2 = static_cast<time_type>(6.0);     // FAC2 (max step size factor)
    static constexpr time_type fortran_beta = static_cast<time_type>(0.0);     // BETA (step size stabilization)
    static constexpr time_type fortran_dt_max = static_cast<time_type>(1e100); // HMAX (max step size)
    static constexpr time_type fortran_dt_min = static_cast<time_type>(1e-16); // practical min (not in Fortran, but practical)
    static constexpr int fortran_nmax = 100000;                                // NMAX (max steps)
    static constexpr int fortran_nstiff = 1000;                                // NSTIFF (stiffness test interval)

    // Internal state (Fortran default values)
    time_type safety_factor_ = fortran_safety;
    time_type fac1_ = fortran_fac1;
    time_type fac2_ = fortran_fac2;
    time_type beta_ = fortran_beta;
    time_type dt_max_ = fortran_dt_max;
    time_type dt_min_ = fortran_dt_min;
    int nmax_ = fortran_nmax;
    int nstiff_ = fortran_nstiff;
    time_type facold_ = static_cast<time_type>(1e-4); // Fortran FACOLD
    // Stiffness detection state
    int iastiff_ = 0;
    int nonsti_ = 0;
    time_type hlamb_ = 0;
    // For statistics (optional)
    int nstep_ = 0;
    int naccpt_ = 0;
    int nrejct_ = 0;
    int nfcn_ = 0;

private:
    void check_nan_inf(const std::string& context, const state_type& state, const state_type& y_new, 
                       const state_type& error, time_type dt, time_type err, time_type err2, time_type deno) {
        // Check for NaN/Inf in all vectors and scalars
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (std::isnan(state[i]) || std::isinf(state[i])) {
                throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " state[" + std::to_string(i) + "]=" + std::to_string(state[i]));
            }
        }
        for (std::size_t i = 0; i < y_new.size(); ++i) {
            if (std::isnan(y_new[i]) || std::isinf(y_new[i])) {
                throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " y_new[" + std::to_string(i) + "]=" + std::to_string(y_new[i]));
            }
        }
        for (std::size_t i = 0; i < error.size(); ++i) {
            if (std::isnan(error[i]) || std::isinf(error[i])) {
                throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " error[" + std::to_string(i) + "]=" + std::to_string(error[i]));
            }
        }
        if (std::isnan(dt) || std::isinf(dt)) {
            throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " dt=" + std::to_string(dt));
        }
        if (std::isnan(err) || std::isinf(err)) {
            throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " err=" + std::to_string(err));
        }
        if (std::isnan(err2) || std::isinf(err2)) {
            throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " err2=" + std::to_string(err2));
        }
        if (std::isnan(deno) || std::isinf(deno)) {
            throw std::runtime_error("DOP853: NaN/Inf detected in " + context + " deno=" + std::to_string(deno));
        }
    }

    // Compute a good initial step size (HINIT from Fortran)
    time_type compute_initial_step(const state_type& y, time_type t, const system_function& sys, time_type t_end) const {
        // Compute f0 = f(t, y)
        state_type f0 = StateCreator<state_type>::create(y);
        sys(t, y, f0);

        // Compute a norm for y and f0
        time_type dnf = 0.0, dny = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            time_type sk = this->atol_ + this->rtol_ * std::abs(y[i]);
            dnf += (f0[i] / sk) * (f0[i] / sk);
            dny += (y[i] / sk) * (y[i] / sk);
        }
        time_type h = 1e-6;
        if (dnf > 1e-10 && dny > 1e-10) {
            h = std::sqrt(dny / dnf) * 0.01;
        }
        h = std::min<time_type>(h, std::abs(t_end - t));
        h = std::copysign(h, t_end - t);

        // Perform an explicit Euler step
        state_type y1 = StateCreator<state_type>::create(y);
        for (std::size_t i = 0; i < y.size(); ++i)
            y1[i] = y[i] + h * f0[i];
        state_type f1 = StateCreator<state_type>::create(y);
        sys(t + h, y1, f1);

        // Estimate the second derivative
        time_type der2 = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            time_type sk = this->atol_ + this->rtol_ * std::abs(y[i]);
            der2 += ((f1[i] - f0[i]) / sk) * ((f1[i] - f0[i]) / sk);
        }
        der2 = std::sqrt(der2) / h;

        // Step size is computed such that h^order * max(norm(f0), norm(der2)) = 0.01
        time_type der12 = std::max<time_type>(std::abs(der2), std::sqrt(dnf));
        time_type h1 = h;
        if (der12 > 1e-15) {
            h1 = std::pow(0.01 / der12, 1.0 / 8.0);
        } else {
            h1 = std::max<time_type>(1e-6, std::abs(h) * 1e-3);
        }
        // Avoid std::min(a, b, c) which is not standard C++
        time_type hmax = 100 * std::abs(h);
        time_type htmp = (h1 < hmax) ? h1 : hmax;
        htmp = (htmp < std::abs(t_end - t)) ? htmp : std::abs(t_end - t);
        h = std::copysign(htmp, t_end - t);
        return h;
    }

public:
    explicit DOP853Integrator(system_function sys, 
                             time_type rtol = static_cast<time_type>(1e-8),
                             time_type atol = static_cast<time_type>(1e-10))
        : base_type(std::move(sys), rtol, atol) {}

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    // To match Fortran DOP853, we need to know the integration target time for HINIT
    // This version assumes you set target_time_ before calling adaptive_step
    time_type target_time_ = 0; // User must set this before integration

    time_type adaptive_step(state_type& state, time_type dt) override {
        // Fortran DOP853: if dt <= 0, estimate initial step size using HINIT (compute_initial_step)
        time_type t = this->current_time_;
        time_type t_end = target_time_;
        if (t_end == t) t_end = t + 1.0; // fallback if not set
        time_type current_dt = dt;
        if (current_dt <= 0) {
            // Use the system function and current state to estimate initial step
            current_dt = compute_initial_step(state, t, this->sys_, t_end);
            // Clamp to allowed min/max
            current_dt = std::max<time_type>(dt_min_, std::min<time_type>(dt_max_, current_dt));
        }
        int attempt = 0;
        for (; attempt < nmax_; ++attempt) {
            
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            dop853_step(state, y_new, error, current_dt);
            nfcn_ += 12; // 12 stages per step

            // Check for NaN/Inf in step computation
            check_nan_inf("step_computation", state, y_new, error, current_dt, 0.0, 0.0, 0.0);

            // Fortran error norm (ERR, ERR2, DENO, etc.)
            time_type err = 0.0, err2 = 0.0;
            for (std::size_t i = 0; i < state.size(); ++i) {
                time_type sk = this->atol_ + this->rtol_ * std::max<time_type>(std::abs(state[i]), std::abs(y_new[i]));
                // Fortran: ERRI=K4(I)-BHH1*K1(I)-BHH2*K9(I)-BHH3*K3(I)  (here, error[i] is 8th-5th order diff)
                // We use error[i] as the embedded error estimate, so for full Fortran, you may need to store all k's
                err2 += (error[i] / sk) * (error[i] / sk); // proxy for Fortran's ERR2
                err += (error[i] / sk) * (error[i] / sk); // Fortran's ERR
            }
            time_type deno = err + 0.01 * err2;
            if (deno <= 0.0 || std::isnan(deno) || std::isinf(deno)) {
                deno = 1.0;
            }
            err = std::abs(current_dt) * err * std::sqrt(1.0 / (state.size() * deno));
            if (std::isnan(err) || std::isinf(err)) {
                err = 1.0;
            }

            // Check for NaN/Inf in error norm calculation
            check_nan_inf("error_norm", state, y_new, error, current_dt, err, err2, deno);

            // Fortran: FAC11 = ERR**EXPO1, FAC = FAC11 / FACOLD**BETA
            time_type expo1 = 1.0 / 8.0 - beta_ * 0.2;
            time_type fac11 = std::pow(std::max<time_type>(err, static_cast<time_type>(1e-16)), expo1);
            time_type fac = fac11 / std::pow(facold_, beta_);
            // Clamp fac between fac1_ (min, <1) and fac2_ (max, >1)
            fac = std::min<time_type>(fac2_, std::max<time_type>(fac1_, fac / safety_factor_));
            if (std::isnan(fac) || std::isinf(fac)) {
                fac = 1.0;
            }
            time_type next_dt = current_dt / fac;
            if (next_dt <= 0.0 || std::isnan(next_dt) || std::isinf(next_dt)) {
                next_dt = dt_min_;
            }

            if (err <= 1.0) {
                facold_ = std::max<time_type>(err, static_cast<time_type>(1e-4));
                naccpt_++;
                nstep_++;
                state = y_new;
                this->advance_time(current_dt);
                
                // stiffness detection (Fortran HLAMB)
                if (nstiff_ > 0 && (naccpt_ % nstiff_ == 0 || iastiff_ > 0)) {
                    // Compute HLAMB = |h| * sqrt(stnum / stden)
                    time_type stnum = 0, stden = 0;
                    for (std::size_t i = 0; i < state.size(); ++i) {
                        stnum += (error[i]) * (error[i]);
                        stden += (y_new[i] - state[i]) * (y_new[i] - state[i]);
                    }
                    if (stden > 0) hlamb_ = std::abs(current_dt) * std::sqrt(stnum / stden);
                    if (hlamb_ > 6.1) {
                        nonsti_ = 0;
                        iastiff_++;
                        if (iastiff_ == 15) {
                            throw std::runtime_error("DOP853: Problem seems to become stiff");
                        }
                    } else {
                        nonsti_++;
                        if (nonsti_ == 6) iastiff_ = 0;
                    }
                }
                // Clamp next step size
                next_dt = std::max<time_type>(dt_min_, std::min<time_type>(dt_max_, next_dt));
                return next_dt;
            } else {
                // Step rejected
                nrejct_++;
                nstep_++;
                next_dt = current_dt / std::min<time_type>(fac1_, fac11 / safety_factor_);
                current_dt = std::max<time_type>(dt_min_, next_dt);
            }
        }
        throw std::runtime_error("DOP853: Maximum number of step size reductions or steps exceeded");
    }

private:
    // Helper functions to access coefficients
    static constexpr time_type get_c(int i) { return diffeq::integrators::ode::dop853::C<time_type>[i]; }
    static constexpr time_type get_a(int i, int j) { return diffeq::integrators::ode::dop853::A<time_type>[i][j]; }
    static constexpr time_type get_b(int i) { return diffeq::integrators::ode::dop853::B<time_type>[i]; }
    static constexpr time_type get_e5(int i) { return diffeq::integrators::ode::dop853::E5<time_type>[i]; }

    void dop853_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Allocate all needed k vectors
        std::vector<state_type> k(12, StateCreator<state_type>::create(y));
        state_type temp = StateCreator<state_type>::create(y);
        time_type t = this->current_time_;

        // k1 = f(t, y)
        this->sys_(t, y, k[0]);

        // k2 = f(t + c2*dt, y + dt*a21*k1)
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * get_a(1, 0) * k[0][i];
        this->sys_(t + get_c(1) * dt, temp, k[1]);

        // k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(2, 0) * k[0][i] + get_a(2, 1) * k[1][i]);
        this->sys_(t + get_c(2) * dt, temp, k[2]);

        // k4 = f(t + c4*dt, y + dt*(a41*k1 + a43*k3))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(3, 0) * k[0][i] + get_a(3, 2) * k[2][i]);
        this->sys_(t + get_c(3) * dt, temp, k[3]);

        // k5 = f(t + c5*dt, y + dt*(a51*k1 + a53*k3 + a54*k4))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(4, 0) * k[0][i] + get_a(4, 2) * k[2][i] + get_a(4, 3) * k[3][i]);
        this->sys_(t + get_c(4) * dt, temp, k[4]);

        // k6 = f(t + c6*dt, y + dt*(a61*k1 + a64*k4 + a65*k5))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(5, 0) * k[0][i] + get_a(5, 3) * k[3][i] + get_a(5, 4) * k[4][i]);
        this->sys_(t + get_c(5) * dt, temp, k[5]);

        // k7 = f(t + c7*dt, y + dt*(a71*k1 + a74*k4 + a75*k5 + a76*k6))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(6, 0) * k[0][i] + get_a(6, 3) * k[3][i] + get_a(6, 4) * k[4][i] + get_a(6, 5) * k[5][i]);
        this->sys_(t + get_c(6) * dt, temp, k[6]);

        // k8 = f(t + c8*dt, y + dt*(a81*k1 + a84*k4 + a85*k5 + a86*k6 + a87*k7))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(7, 0) * k[0][i] + get_a(7, 3) * k[3][i] + get_a(7, 4) * k[4][i] + get_a(7, 5) * k[5][i] + get_a(7, 6) * k[6][i]);
        this->sys_(t + get_c(7) * dt, temp, k[7]);

        // k9 = f(t + c9*dt, y + dt*(a91*k1 + a94*k4 + a95*k5 + a96*k6 + a97*k7 + a98*k8))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(8, 0) * k[0][i] + get_a(8, 3) * k[3][i] + get_a(8, 4) * k[4][i] + get_a(8, 5) * k[5][i] + get_a(8, 6) * k[6][i] + get_a(8, 7) * k[7][i]);
        this->sys_(t + get_c(8) * dt, temp, k[8]);

        // k10 = f(t + c10*dt, y + dt*(a101*k1 + a104*k4 + a105*k5 + a106*k6 + a107*k7 + a108*k8 + a109*k9))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(9, 0) * k[0][i] + get_a(9, 3) * k[3][i] + get_a(9, 4) * k[4][i] + get_a(9, 5) * k[5][i] + get_a(9, 6) * k[6][i] + get_a(9, 7) * k[7][i] + get_a(9, 8) * k[8][i]);
        this->sys_(t + get_c(9) * dt, temp, k[9]);

        // k11 = f(t + c11*dt, y + dt*(a111*k1 + a114*k4 + a115*k5 + a116*k6 + a117*k7 + a118*k8 + a119*k9 + a1110*k10))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(10, 0) * k[0][i] + get_a(10, 3) * k[3][i] + get_a(10, 4) * k[4][i] + get_a(10, 5) * k[5][i] + get_a(10, 6) * k[6][i] + get_a(10, 7) * k[7][i] + get_a(10, 8) * k[8][i] + get_a(10, 9) * k[9][i]);
        this->sys_(t + get_c(10) * dt, temp, k[10]);

        // k12 = f(t + dt, y + dt*(a121*k1 + a124*k4 + a125*k5 + a126*k6 + a127*k7 + a128*k8 + a129*k9 + a1210*k10 + a1211*k11))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (get_a(11, 0) * k[0][i] + get_a(11, 3) * k[3][i] + get_a(11, 4) * k[4][i] + get_a(11, 5) * k[5][i] + get_a(11, 6) * k[6][i] + get_a(11, 7) * k[7][i] + get_a(11, 8) * k[8][i] + get_a(11, 9) * k[9][i] + get_a(11, 10) * k[10][i]);
        this->sys_(t + dt, temp, k[11]);

        // 8th order solution (y_new)
        for (std::size_t i = 0; i < y.size(); ++i) {
            y_new[i] = y[i] + dt * (get_b(0) * k[0][i] + get_b(5) * k[5][i] + get_b(6) * k[6][i] + get_b(7) * k[7][i] + get_b(8) * k[8][i] + get_b(9) * k[9][i] + get_b(10) * k[10][i] + get_b(11) * k[11][i]);
        }

        // 5th order error estimate (embedded)
        for (std::size_t i = 0; i < y.size(); ++i) {
            error[i] = dt * (get_e5(0) * k[0][i] + get_e5(5) * k[5][i] + get_e5(6) * k[6][i] + get_e5(7) * k[7][i] + get_e5(8) * k[8][i] + get_e5(9) * k[9][i] + get_e5(10) * k[10][i] + get_e5(11) * k[11][i]);
        }
    }
};

} // namespace diffeq
