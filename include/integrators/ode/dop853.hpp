#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <stdexcept>

namespace diffeq::integrators::ode {

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

    explicit DOP853Integrator(system_function sys, 
                             time_type rtol = static_cast<time_type>(1e-8),
                             time_type atol = static_cast<time_type>(1e-10))
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
            dop853_step(state, y_new, error, current_dt);
            time_type err_norm = this->error_norm(error, y_new);
            if (err_norm <= 1.0) {
                state = y_new;
                this->advance_time(current_dt);
                time_type next_dt = this->suggest_step_size(current_dt, err_norm, 8);
                return std::max(this->dt_min_, std::min(this->dt_max_, next_dt));
            } else {
                current_dt *= std::max(this->safety_factor_ * std::pow(err_norm, -1.0/8.0), static_cast<time_type>(0.1));
                current_dt = std::max(current_dt, this->dt_min_);
            }
        }
        throw std::runtime_error("DOP853: Maximum number of step size reductions exceeded");
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
