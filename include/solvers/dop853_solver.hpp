#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <array>
#include <algorithm>

// DOP853 integrator - High-order Runge-Kutta method with adaptive step size control
// Based on scipy implementation: https://github.com/scipy/scipy/blob/v1.16.0/scipy/integrate/_ivp/rk.py
template<typename S, typename T = double>
class DOP853Integrator : public AdaptiveIntegrator<S, T> {
public:
    using base_type = AdaptiveIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    static constexpr int N_STAGES = 12;

    explicit DOP853Integrator(system_function sys, 
                             time_type rtol = static_cast<time_type>(1e-3),
                             time_type atol = static_cast<time_type>(1e-6))
        : base_type(std::move(sys), rtol, atol), 
          safety_(static_cast<time_type>(0.9)),
          min_factor_(static_cast<time_type>(0.2)),
          max_factor_(static_cast<time_type>(10.0)) {
        this->dt_min_ = static_cast<time_type>(1e-12);
        this->dt_max_ = static_cast<time_type>(1e6);
        initialize_coefficients();
    }

    void step(state_type& state, time_type dt) override {
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        time_type h_abs = std::abs(dt);
        time_type min_step = 10 * std::abs(std::nextafter(this->current_time_, 
                            this->current_time_ + dt) - this->current_time_);
        
        if (h_abs > this->dt_max_) {
            h_abs = this->dt_max_;
        } else if (h_abs < min_step) {
            h_abs = min_step;
        }
        
        bool step_accepted = false;
        bool step_rejected = false;
        time_type actual_dt = 0;
        
        while (!step_accepted) {
            if (h_abs < min_step) {
                throw std::runtime_error("Step size too small in DOP853");
            }
            
            time_type h = h_abs * (dt >= 0 ? 1 : -1);
            
            state_type y_new = StateCreator<state_type>::create(state);
            time_type error_norm = rk_step(state, y_new, h);
            
            // Compute scale for error estimation
            time_type scale_norm = 0;
            std::size_t n = state.size();
            for (std::size_t i = 0; i < n; ++i) {
                auto state_it = state.begin();
                auto y_new_it = y_new.begin();
                time_type scale = this->atol_ + std::max(std::abs(state_it[i]), std::abs(y_new_it[i])) * this->rtol_;
                scale_norm += 1.0 / (scale * scale);
            }
            scale_norm = std::sqrt(scale_norm / n);
            error_norm *= scale_norm;
            
            if (error_norm < 1) {
                time_type factor;
                if (error_norm == 0) {
                    factor = max_factor_;
                } else {
                    factor = std::min(max_factor_, safety_ * std::pow(error_norm, error_exponent_));
                }
                
                if (step_rejected) {
                    factor = std::min(static_cast<time_type>(1), factor);
                }
                
                h_abs *= factor;
                step_accepted = true;
                actual_dt = h;
                
                // Update state
                state = y_new;
                this->advance_time(actual_dt);
                
            } else {
                h_abs *= std::max(min_factor_, safety_ * std::pow(error_norm, error_exponent_));
                step_rejected = true;
            }
        }
        
        return actual_dt;
    }

private:
    std::array<time_type, N_STAGES + 1> C_;
    std::array<std::array<time_type, N_STAGES>, N_STAGES> A_;
    std::array<time_type, N_STAGES> B_;
    std::array<time_type, N_STAGES + 1> E3_;  // 3rd order error estimate
    std::array<time_type, N_STAGES + 1> E5_;  // 5th order error estimate
    
    // Adaptive step control parameters
    time_type safety_;
    time_type min_factor_;
    time_type max_factor_;
    static constexpr time_type error_exponent_ = -1.0 / 8.0;  // -1/(order+1) for 7th order error estimator
    
    void initialize_coefficients() {
        // C coefficients (times for stages)
        C_[0] = static_cast<time_type>(0.0);
        C_[1] = static_cast<time_type>(0.526001519587677318785587544488e-01);
        C_[2] = static_cast<time_type>(0.789002279381515978178381316732e-01);
        C_[3] = static_cast<time_type>(0.118350341907227396726757197510);
        C_[4] = static_cast<time_type>(0.281649658092772603273242802490);
        C_[5] = static_cast<time_type>(0.333333333333333333333333333333);
        C_[6] = static_cast<time_type>(0.25);
        C_[7] = static_cast<time_type>(0.307692307692307692307692307692);
        C_[8] = static_cast<time_type>(0.651282051282051282051282051282);
        C_[9] = static_cast<time_type>(0.6);
        C_[10] = static_cast<time_type>(0.857142857142857142857142857142);
        C_[11] = static_cast<time_type>(1.0);
        C_[12] = static_cast<time_type>(1.0);

        // Initialize A matrix to zero
        for (int i = 0; i < N_STAGES; ++i) {
            for (int j = 0; j < N_STAGES; ++j) {
                A_[i][j] = static_cast<time_type>(0);
            }
        }

        // A matrix coefficients (complete DOP853 coefficients from SciPy)
        A_[1][0] = static_cast<time_type>(5.26001519587677318785587544488e-2);
        
        A_[2][0] = static_cast<time_type>(1.97250569845378994544595329183e-2);
        A_[2][1] = static_cast<time_type>(5.91751709536136983633785987549e-2);
        
        A_[3][0] = static_cast<time_type>(2.95875854768068491816892993775e-2);
        A_[3][2] = static_cast<time_type>(8.87627564304205475450678981324e-2);
        
        A_[4][0] = static_cast<time_type>(2.41365134159266685502369798665e-1);
        A_[4][2] = static_cast<time_type>(-8.84549479328286085344864962717e-1);
        A_[4][3] = static_cast<time_type>(9.24834003261792003115737966543e-1);
        
        A_[5][0] = static_cast<time_type>(3.7037037037037037037037037037e-2);
        A_[5][3] = static_cast<time_type>(1.70828608729473871279604482173e-1);
        A_[5][4] = static_cast<time_type>(1.25467687566822425016691814123e-1);
        
        A_[6][0] = static_cast<time_type>(3.7109375e-2);
        A_[6][3] = static_cast<time_type>(1.70252211019544039314978060272e-1);
        A_[6][4] = static_cast<time_type>(6.02165389804559606850219397283e-2);
        A_[6][5] = static_cast<time_type>(-1.7578125e-2);
        
        A_[7][0] = static_cast<time_type>(3.70920001185047927108779319836e-2);
        A_[7][3] = static_cast<time_type>(1.70383925712239993810214054705e-1);
        A_[7][4] = static_cast<time_type>(1.07262030446373284651809199168e-1);
        A_[7][5] = static_cast<time_type>(-1.53194377486244017527936158236e-2);
        A_[7][6] = static_cast<time_type>(8.27378916381402288758473766002e-3);
        
        A_[8][0] = static_cast<time_type>(6.24110958716075717114429577812e-1);
        A_[8][3] = static_cast<time_type>(-3.36089262944694129406857109825);
        A_[8][4] = static_cast<time_type>(-8.68219346841726006818189891453e-1);
        A_[8][5] = static_cast<time_type>(2.75920996994467083049415600797e1);
        A_[8][6] = static_cast<time_type>(2.01540675504778934086186788979e1);
        A_[8][7] = static_cast<time_type>(-4.34898841810699588477366255144e1);
        
        A_[9][0] = static_cast<time_type>(4.77662536438264365890433908527e-1);
        A_[9][3] = static_cast<time_type>(-2.48811461997166764192642586468);
        A_[9][4] = static_cast<time_type>(-5.90290826836842996371446475743e-1);
        A_[9][5] = static_cast<time_type>(2.12300514481811942347288949897e1);
        A_[9][6] = static_cast<time_type>(1.52792336328824235832596922938e1);
        A_[9][7] = static_cast<time_type>(-3.32882109689848629194453265587e1);
        A_[9][8] = static_cast<time_type>(-2.03312017085086261358222928593e-2);
        
        A_[10][0] = static_cast<time_type>(-9.3714243008598732571704021658e-1);
        A_[10][3] = static_cast<time_type>(5.18637242884406370830023853209);
        A_[10][4] = static_cast<time_type>(1.09143734899672957818500254654);
        A_[10][5] = static_cast<time_type>(-8.14978701074692612513997267357);
        A_[10][6] = static_cast<time_type>(-1.85200656599969598641566180701e1);
        A_[10][7] = static_cast<time_type>(2.27394870993505042818970056734e1);
        A_[10][8] = static_cast<time_type>(2.49360555267965238987089396762);
        A_[10][9] = static_cast<time_type>(-3.0467644718982195003823669022);
        
        A_[11][0] = static_cast<time_type>(2.27331014751653820792359768449);
        A_[11][3] = static_cast<time_type>(-1.05344954667372501984066689879e1);
        A_[11][4] = static_cast<time_type>(-2.00087205822486249909675718444);
        A_[11][5] = static_cast<time_type>(-1.79589318631187989172765950534e1);
        A_[11][6] = static_cast<time_type>(2.79488845294199600508499808837e1);
        A_[11][7] = static_cast<time_type>(-2.85899827713502369474065508674);
        A_[11][8] = static_cast<time_type>(-8.87285693353062954433549289258);
        A_[11][9] = static_cast<time_type>(1.23605671757943030647266201528e1);
        A_[11][10] = static_cast<time_type>(6.43392746015763530355970484046e-1);

        // B coefficients (8th order solution)
        B_[0] = static_cast<time_type>(5.42937341165687622380535766363e-2);
        B_[1] = static_cast<time_type>(0.0);
        B_[2] = static_cast<time_type>(0.0);
        B_[3] = static_cast<time_type>(0.0);
        B_[4] = static_cast<time_type>(0.0);
        B_[5] = static_cast<time_type>(4.45031289275240888144113950566);
        B_[6] = static_cast<time_type>(1.89151789931450038304281599044);
        B_[7] = static_cast<time_type>(-5.8012039600105847814672114227);
        B_[8] = static_cast<time_type>(3.1116436695781989440891606237e-1);
        B_[9] = static_cast<time_type>(-1.52160949662516078556178806805e-1);
        B_[10] = static_cast<time_type>(2.01365400804030348374776537501e-1);
        B_[11] = static_cast<time_type>(4.47106157277725905176885569043e-2);
        
        // E3 coefficients (3rd order error estimate)
        E3_[0] = B_[0] - static_cast<time_type>(0.244094488188976377952755905512);
        E3_[1] = B_[1];
        E3_[2] = B_[2];
        E3_[3] = B_[3];
        E3_[4] = B_[4];
        E3_[5] = B_[5];
        E3_[6] = B_[6];
        E3_[7] = B_[7];
        E3_[8] = B_[8] - static_cast<time_type>(0.733846688281611857341361741547);
        E3_[9] = B_[9];
        E3_[10] = B_[10];
        E3_[11] = B_[11] - static_cast<time_type>(0.220588235294117647058823529412e-1);
        E3_[12] = static_cast<time_type>(0.0);
        
        // E5 coefficients (5th order error estimate)
        E5_[0] = static_cast<time_type>(0.1312004499419488073250102996e-1);
        E5_[1] = static_cast<time_type>(0.0);
        E5_[2] = static_cast<time_type>(0.0);
        E5_[3] = static_cast<time_type>(0.0);
        E5_[4] = static_cast<time_type>(0.0);
        E5_[5] = static_cast<time_type>(-0.1225156446376204440720569753e+1);
        E5_[6] = static_cast<time_type>(-0.4957589496572501915214079952);
        E5_[7] = static_cast<time_type>(0.1664377182454986536961530415e+1);
        E5_[8] = static_cast<time_type>(-0.3503288487499736816886487290);
        E5_[9] = static_cast<time_type>(0.3341791187130174790297318841);
        E5_[10] = static_cast<time_type>(0.8192320648511571246570742613e-1);
        E5_[11] = static_cast<time_type>(-0.2235530786388629525884427845e-1);
        E5_[12] = static_cast<time_type>(0.0);
    }

    
    // Core RK step implementation following scipy's rk_step function
    time_type rk_step(const state_type& y, state_type& y_new, time_type h) {
        std::array<state_type, N_STAGES + 1> K;
        for (int i = 0; i <= N_STAGES; ++i) {
            K[i] = StateCreator<state_type>::create(y);
        }
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        
        // Stage 1: K[0] = f(t, y)
        this->sys_(t, y, K[0]);
        
        // Compute remaining stages
        for (int stage = 1; stage < N_STAGES; ++stage) {
            // temp = y + h * sum(A[stage][j] * K[j] for j in range(stage))
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i];
                
                for (int j = 0; j < stage; ++j) {
                    if (A_[stage][j] != static_cast<time_type>(0)) {
                        auto k_it = K[j].begin();
                        temp_it[i] += h * A_[stage][j] * k_it[i];
                    }
                }
            }
            
            this->sys_(t + C_[stage] * h, temp, K[stage]);
        }
        
        // Compute final solution: y_new = y + h * sum(B[j] * K[j])
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto y_new_it = y_new.begin();
            y_new_it[i] = y_it[i];
            
            for (int j = 0; j < N_STAGES; ++j) {
                if (B_[j] != static_cast<time_type>(0)) {
                    auto k_it = K[j].begin();
                    y_new_it[i] += h * B_[j] * k_it[i];
                }
            }
        }
        
        // Compute final derivative for next step
        this->sys_(t + h, y_new, K[N_STAGES]);
        
        // Estimate error using combination of 5th and 3rd order estimates (scipy style)
        time_type err5_norm_2 = 0;
        time_type err3_norm_2 = 0;
        std::size_t n = y.size();
        
        for (std::size_t i = 0; i < n; ++i) {
            time_type err5 = 0;
            time_type err3 = 0;
            
            for (int j = 0; j <= N_STAGES; ++j) {
                auto k_it = K[j].begin();
                err5 += E5_[j] * k_it[i];
                err3 += E3_[j] * k_it[i];
            }
            
            err5_norm_2 += err5 * err5;
            err3_norm_2 += err3 * err3;
        }
        
        if (err5_norm_2 == 0 && err3_norm_2 == 0) {
            return 0.0;
        }
        
        time_type denom = err5_norm_2 + 0.01 * err3_norm_2;
        return std::abs(h) * err5_norm_2 / std::sqrt(denom * n);
    }
};
