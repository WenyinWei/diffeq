
#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <stdexcept>

namespace diffeq::integrators::ode {

template<system_state S, can_be_time T>
class DOP853Integrator;

template<system_state S, can_be_time T>
class DOP853DenseOutputHelper {
public:
    using value_type = typename DOP853Integrator<S, T>::value_type;
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
template<system_state S, can_be_time T = double>
class DOP853Integrator : public AdaptiveIntegrator<S, T> {
    
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    // Step size control parameters (modern C++ style, with defaults)
    time_type safety_factor_ = static_cast<time_type>(0.9);   // Fortran SAFE
    time_type fac1_ = static_cast<time_type>(0.333);          // Fortran FAC1 (min step size factor)
    time_type fac2_ = static_cast<time_type>(6.0);            // Fortran FAC2 (max step size factor)
    time_type beta_ = static_cast<time_type>(0.0);            // Fortran BETA (step size stabilization)
    time_type dt_max_ = static_cast<time_type>(1e100);        // Fortran HMAX (max step size)
    time_type dt_min_ = static_cast<time_type>(1e-16);        // Min step size (not in Fortran, but practical)
    int nmax_ = 100000;                                       // Fortran NMAX (max steps)
    int nstiff_ = 1000;                                       // Fortran NSTIFF (stiffness test interval)

    // Stiffness detection state
    int iastiff_ = 0;
    int nonsti_ = 0;
    time_type hlamb_ = 0;
    time_type facold_ = static_cast<time_type>(1e-4);

    // For statistics (optional)
    int nstep_ = 0;
    int naccpt_ = 0;
    int nrejct_ = 0;
    int nfcn_ = 0;

private:
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
        h = std::min(h, std::abs(t_end - t));
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
        time_type der12 = std::max(std::abs(der2), std::sqrt(dnf));
        time_type h1 = h;
        if (der12 > 1e-15) {
            h1 = std::pow(0.01 / der12, 1.0 / 8.0);
        } else {
            h1 = std::max(1e-6, std::abs(h) * 1e-3);
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
                             time_type atol = static_cast<time_type>(1e-10),
                             time_type safety = static_cast<time_type>(0.9),
                             time_type fac1 = static_cast<time_type>(0.333),
                             time_type fac2 = static_cast<time_type>(6.0),
                             time_type beta = static_cast<time_type>(0.0),
                             time_type dt_max = static_cast<time_type>(1e100),
                             time_type dt_min = static_cast<time_type>(1e-16),
                             int nmax = 100000,
                             int nstiff = 1000)
        : base_type(std::move(sys), rtol, atol),
          safety_factor_(safety), fac1_(fac1), fac2_(fac2), beta_(beta),
          dt_max_(dt_max), dt_min_(dt_min), nmax_(nmax), nstiff_(nstiff) {}

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
            current_dt = std::max(dt_min_, std::min(dt_max_, current_dt));
        }
        int attempt = 0;
        for (; attempt < nmax_; ++attempt) {
            state_type y_new = StateCreator<state_type>::create(state);
            state_type error = StateCreator<state_type>::create(state);
            dop853_step(state, y_new, error, current_dt);
            nfcn_ += 12; // 12 stages per step
            time_type err_norm = this->error_norm(error, y_new);

            // Fortran: FAC11 = ERR**EXPO1, FAC = FAC11 / FACOLD**BETA
            time_type expo1 = 1.0 / 8.0 - beta_ * 0.2;
            time_type fac11 = std::pow(std::max(err_norm, 1e-16), expo1);
            time_type fac = fac11 / std::pow(facold_, beta_);
            fac = std::max(fac2_, std::min(fac1_, fac / safety_factor_));
            time_type next_dt = current_dt / fac;

            if (err_norm <= 1.0) {
                facold_ = std::max(err_norm, static_cast<time_type>(1e-4));
                naccpt_++;
                nstep_++;
                state = y_new;
                this->advance_time(current_dt);
                // stiffness detection (Fortran HLAMB)
                if (nstiff_ > 0 && (naccpt_ % nstiff_ == 0 || iastiff_ > 0)) {
                    // Compute HLAMB = |h| * sqrt(stnum / stden)
                    time_type stnum = 0, stden = 0;
                    for (std::size_t i = 0; i < state.size(); ++i) {
                        // Use error and state difference as proxy (not exact Fortran, but close)
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
                next_dt = std::max(dt_min_, std::min(dt_max_, next_dt));
                // Accept and return next step size
                return next_dt;
            } else {
                // Step rejected
                nrejct_++;
                nstep_++;
                next_dt = current_dt / std::min(fac1_, fac11 / safety_factor_);
                current_dt = std::max(dt_min_, next_dt);
            }
        }
        throw std::runtime_error("DOP853: Maximum number of step size reductions or steps exceeded");
    }

private:
    // DOP853 Butcher tableau coefficients (from Fortran code)
    static constexpr time_type c2  = 0.0526001519587677318785587544488;
    static constexpr time_type c3  = 0.0789002279381515978178381316732;
    static constexpr time_type c4  = 0.118350341907227396726757197510;
    static constexpr time_type c5  = 0.281649658092772603273242802490;
    static constexpr time_type c6  = 0.333333333333333333333333333333;
    static constexpr time_type c7  = 0.25;
    static constexpr time_type c8  = 0.307692307692307692307692307692;
    static constexpr time_type c9  = 0.651282051282051282051282051282;
    static constexpr time_type c10 = 0.6;
    static constexpr time_type c11 = 0.857142857142857142857142857142;

    // a_ij coefficients (only nonzero, as in Fortran)
    static constexpr time_type a21 = 0.0526001519587677318785587544488;
    static constexpr time_type a31 = 0.0197250569845378994544595329183;
    static constexpr time_type a32 = 0.0591751709536136983633785987549;
    static constexpr time_type a41 = 0.0295875854768068491816892993775;
    static constexpr time_type a43 = 0.0887627564304205475450678981324;
    static constexpr time_type a51 = 0.241365134159266685502369798665;
    static constexpr time_type a53 = -0.884549479328286085344864962717;
    static constexpr time_type a54 = 0.924834003261792003115737966543;
    static constexpr time_type a61 = 0.037037037037037037037037037037;
    static constexpr time_type a64 = 0.170828608729473871279604482173;
    static constexpr time_type a65 = 0.125467687566822425016691814123;
    static constexpr time_type a71 = 0.037109375;
    static constexpr time_type a74 = 0.170252211019544039314978060272;
    static constexpr time_type a75 = 0.0602165389804559606850219397283;
    static constexpr time_type a76 = -0.017578125;
    static constexpr time_type a81 = 0.0370920001185047927108779319836;
    static constexpr time_type a84 = 0.170383925712239993810214054705;
    static constexpr time_type a85 = 0.107262030446373284651809199168;
    static constexpr time_type a86 = -0.0153194377486244017527936158236;
    static constexpr time_type a87 = 0.00827378916381402288758473766002;
    static constexpr time_type a91 = 0.624110958716075717114429577812;
    static constexpr time_type a94 = -3.36089262944694129406857109825;
    static constexpr time_type a95 = -0.868219346841726006818189891453;
    static constexpr time_type a96 = 27.5920996994467083049415600797;
    static constexpr time_type a97 = 20.1540675504778934086186788979;
    static constexpr time_type a98 = -43.4898841810699588477366255144;
    static constexpr time_type a101 = 0.477662536438264365890433908527;
    static constexpr time_type a104 = -2.48811461997166764192642586468;
    static constexpr time_type a105 = -0.590290826836842996371446475743;
    static constexpr time_type a106 = 21.2300514481811942347288949897;
    static constexpr time_type a107 = 15.2792336328824235832596922938;
    static constexpr time_type a108 = -33.2882109689848629194453265587;
    static constexpr time_type a109 = -0.0203312017085086261358222928593;
    static constexpr time_type a111 = -0.93714243008598732571704021658;
    static constexpr time_type a114 = 5.18637242884406370830023853209;
    static constexpr time_type a115 = 1.09143734899672957818500254654;
    static constexpr time_type a116 = -8.14978701074692612513997267357;
    static constexpr time_type a117 = -18.5200656599969598641566180701;
    static constexpr time_type a118 = 22.7394870993505042818970056734;
    static constexpr time_type a119 = 2.49360555267965238987089396762;
    static constexpr time_type a1110 = -3.0467644718982195003823669022;
    static constexpr time_type a121 = 2.27331014751653820792359768449;
    static constexpr time_type a124 = -10.5344954667372501984066689879;
    static constexpr time_type a125 = -2.00087205822486249909675718444;
    static constexpr time_type a126 = -17.9589318631187989172765950534;
    static constexpr time_type a127 = 27.9488845294199600508499808837;
    static constexpr time_type a128 = -2.85899827713502369474065508674;
    static constexpr time_type a129 = -8.87285693353062954433549289258;
    static constexpr time_type a1210 = 12.3605671757943030647266201528;
    static constexpr time_type a1211 = 0.643392746015763530355970484046;

    // b_i coefficients (8th order solution)
    static constexpr time_type b1 = 0.0542937341165687622380535766363;
    static constexpr time_type b6 = 4.45031289275240888144113950566;
    static constexpr time_type b7 = 1.89151789931450038304281599044;
    static constexpr time_type b8 = -5.8012039600105847814672114227;
    static constexpr time_type b9 = 0.31116436695781989440891606237;
    static constexpr time_type b10 = -0.152160949662516078556178806805;
    static constexpr time_type b11 = 0.201365400804030348374776537501;
    static constexpr time_type b12 = 0.0447106157277725905176885569043;

    // Error coefficients (5th order)
    static constexpr time_type er1 = 0.01312004499419488073250102996;
    static constexpr time_type er6 = -12.25156446376204440720569753;
    static constexpr time_type er7 = -4.957589496572501915214079952;
    static constexpr time_type er8 = 16.64377182454986536961530415;
    static constexpr time_type er9 = -0.3503288487499736816886487290;
    static constexpr time_type er10 = 0.3341791187130174790297318841;
    static constexpr time_type er11 = 0.08192320648511571246570742613;
    static constexpr time_type er12 = -0.02235530786388629525884427845;

    // For brevity, d coefficients for dense output are omitted here

    void dop853_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        // Allocate all needed k vectors
        std::vector<state_type> k(12, StateCreator<state_type>::create(y));
        state_type temp = StateCreator<state_type>::create(y);
        time_type t = this->current_time_;

        // k1 = f(t, y)
        this->sys_(t, y, k[0]);

        // k2 = f(t + c2*dt, y + dt*a21*k1)
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * a21 * k[0][i];
        this->sys_(t + c2 * dt, temp, k[1]);

        // k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a31 * k[0][i] + a32 * k[1][i]);
        this->sys_(t + c3 * dt, temp, k[2]);

        // k4 = f(t + c4*dt, y + dt*(a41*k1 + a43*k3))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a41 * k[0][i] + a43 * k[2][i]);
        this->sys_(t + c4 * dt, temp, k[3]);

        // k5 = f(t + c5*dt, y + dt*(a51*k1 + a53*k3 + a54*k4))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a51 * k[0][i] + a53 * k[2][i] + a54 * k[3][i]);
        this->sys_(t + c5 * dt, temp, k[4]);

        // k6 = f(t + c6*dt, y + dt*(a61*k1 + a64*k4 + a65*k5))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a61 * k[0][i] + a64 * k[3][i] + a65 * k[4][i]);
        this->sys_(t + c6 * dt, temp, k[5]);

        // k7 = f(t + c7*dt, y + dt*(a71*k1 + a74*k4 + a75*k5 + a76*k6))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a71 * k[0][i] + a74 * k[3][i] + a75 * k[4][i] + a76 * k[5][i]);
        this->sys_(t + c7 * dt, temp, k[6]);

        // k8 = f(t + c8*dt, y + dt*(a81*k1 + a84*k4 + a85*k5 + a86*k6 + a87*k7))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a81 * k[0][i] + a84 * k[3][i] + a85 * k[4][i] + a86 * k[5][i] + a87 * k[6][i]);
        this->sys_(t + c8 * dt, temp, k[7]);

        // k9 = f(t + c9*dt, y + dt*(a91*k1 + a94*k4 + a95*k5 + a96*k6 + a97*k7 + a98*k8))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a91 * k[0][i] + a94 * k[3][i] + a95 * k[4][i] + a96 * k[5][i] + a97 * k[6][i] + a98 * k[7][i]);
        this->sys_(t + c9 * dt, temp, k[8]);

        // k10 = f(t + c10*dt, y + dt*(a101*k1 + a104*k4 + a105*k5 + a106*k6 + a107*k7 + a108*k8 + a109*k9))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a101 * k[0][i] + a104 * k[3][i] + a105 * k[4][i] + a106 * k[5][i] + a107 * k[6][i] + a108 * k[7][i] + a109 * k[8][i]);
        this->sys_(t + c10 * dt, temp, k[9]);

        // k11 = f(t + c11*dt, y + dt*(a111*k1 + a114*k4 + a115*k5 + a116*k6 + a117*k7 + a118*k8 + a119*k9 + a1110*k10))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a111 * k[0][i] + a114 * k[3][i] + a115 * k[4][i] + a116 * k[5][i] + a117 * k[6][i] + a118 * k[7][i] + a119 * k[8][i] + a1110 * k[9][i]);
        this->sys_(t + c11 * dt, temp, k[10]);

        // k12 = f(t + dt, y + dt*(a121*k1 + a124*k4 + a125*k5 + a126*k6 + a127*k7 + a128*k8 + a129*k9 + a1210*k10 + a1211*k11))
        for (std::size_t i = 0; i < y.size(); ++i)
            temp[i] = y[i] + dt * (a121 * k[0][i] + a124 * k[3][i] + a125 * k[4][i] + a126 * k[5][i] + a127 * k[6][i] + a128 * k[7][i] + a129 * k[8][i] + a1210 * k[9][i] + a1211 * k[10][i]);
        this->sys_(t + dt, temp, k[11]);

        // 8th order solution (y_new)
        for (std::size_t i = 0; i < y.size(); ++i) {
            y_new[i] = y[i] + dt * (b1 * k[0][i] + b6 * k[5][i] + b7 * k[6][i] + b8 * k[7][i] + b9 * k[8][i] + b10 * k[9][i] + b11 * k[10][i] + b12 * k[11][i]);
        }

        // 5th order error estimate (embedded)
        for (std::size_t i = 0; i < y.size(); ++i) {
            error[i] = dt * (er1 * k[0][i] + er6 * k[5][i] + er7 * k[6][i] + er8 * k[7][i] + er9 * k[8][i] + er10 * k[9][i] + er11 * k[10][i] + er12 * k[11][i]);
        }
    }
};

} // namespace diffeq::integrators::ode
