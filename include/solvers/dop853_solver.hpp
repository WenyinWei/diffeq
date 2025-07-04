#pragma once
#include <core/adaptive_integrator.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <array>
#include <iostream>

// DOP853 integrator - Simplified debug version
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
                             time_type rtol = static_cast<time_type>(1e-6),
                             time_type atol = static_cast<time_type>(1e-9))
        : base_type(std::move(sys), rtol, atol) {
        this->dt_min_ = static_cast<time_type>(1e-10);
        this->dt_max_ = static_cast<time_type>(1e2);
        initialize_coefficients();
        std::cout << "DEBUG: DOP853 constructor completed" << std::endl;
    }

    void step(state_type& state, time_type dt) override {
        std::cout << "DEBUG: DOP853::step called with dt=" << dt << std::endl;
        adaptive_step(state, dt);
    }

    time_type adaptive_step(state_type& state, time_type dt) override {
        std::cout << "DEBUG: adaptive_step called, dt=" << dt << std::endl;
        std::cout << "DEBUG: Current state y[0]=" << *state.begin() << std::endl;
        std::cout << "DEBUG: Current time=" << this->current_time_ << std::endl;
        
        state_type y_new = StateCreator<state_type>::create(state);
        state_type error = StateCreator<state_type>::create(state);
        
        std::cout << "DEBUG: About to call dop853_step..." << std::endl;
        dop853_step(state, y_new, error, dt);
        
        std::cout << "DEBUG: dop853_step completed" << std::endl;
        std::cout << "DEBUG: New state y_new[0]=" << *y_new.begin() << std::endl;
        
        state = y_new;
        this->advance_time(dt);
        
        std::cout << "DEBUG: State updated, new time=" << this->current_time_ << std::endl;
        return dt;
    }

private:
    std::array<time_type, N_STAGES + 1> C_;
    std::array<std::array<time_type, N_STAGES>, N_STAGES> A_;
    std::array<time_type, N_STAGES> B_;
    
    void initialize_coefficients() {
        // C coefficients
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

        // A matrix coefficients (first few only for debugging)
        A_[1][0] = static_cast<time_type>(5.26001519587677318785587544488e-2);
        A_[2][0] = static_cast<time_type>(1.97250569845378994544595329183e-2);
        A_[2][1] = static_cast<time_type>(5.91751709536136983633785987549e-2);
        A_[3][0] = static_cast<time_type>(2.95875854768068491816892993775e-2);
        A_[3][2] = static_cast<time_type>(8.87627564304205475450678981324e-2);

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
    }

    void dop853_step(const state_type& y, state_type& y_new, state_type& error, time_type dt) {
        std::cout << "DEBUG: dop853_step called, dt=" << dt << std::endl;
        std::cout << "DEBUG: Input y[0]=" << *y.begin() << std::endl;
        
        std::array<state_type, N_STAGES> K;
        for (int i = 0; i < N_STAGES; ++i) {
            K[i] = StateCreator<state_type>::create(y);
        }
        state_type temp = StateCreator<state_type>::create(y);
        
        time_type t = this->current_time_;
        std::cout << "DEBUG: Using time t=" << t << std::endl;
        
        // Stage 1: K[0] = f(t, y)
        std::cout << "DEBUG: Calling sys_ for stage 1..." << std::endl;
        this->sys_(t, y, K[0]);
        std::cout << "DEBUG: Stage 1 complete, K[0][0]=" << *K[0].begin() << std::endl;
        
        // For debugging, let's just do the first few stages
        for (int stage = 1; stage < 4; ++stage) {
            std::cout << "DEBUG: Computing stage " << (stage+1) << std::endl;
            
            for (std::size_t i = 0; i < y.size(); ++i) {
                auto y_it = y.begin();
                auto temp_it = temp.begin();
                temp_it[i] = y_it[i];
                
                for (int j = 0; j < stage; ++j) {
                    if (A_[stage][j] != static_cast<time_type>(0)) {
                        auto k_it = K[j].begin();
                        temp_it[i] += dt * A_[stage][j] * k_it[i];
                    }
                }
            }
            
            std::cout << "DEBUG: Calling sys_ for stage " << (stage+1) << std::endl;
            this->sys_(t + C_[stage] * dt, temp, K[stage]);
            std::cout << "DEBUG: Stage " << (stage+1) << " complete" << std::endl;
        }
        
        std::cout << "DEBUG: Computing final solution..." << std::endl;
        
        // Calculate solution using just first few K values for debugging
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto y_it = y.begin();
            auto y_new_it = y_new.begin();
            y_new_it[i] = y_it[i];
            
            // Just use first stage for now
            auto k_it = K[0].begin();
            y_new_it[i] += dt * B_[0] * k_it[i];
        }
        
        std::cout << "DEBUG: Final y_new[0]=" << *y_new.begin() << std::endl;
        
        // Simple error estimate
        for (std::size_t i = 0; i < y.size(); ++i) {
            auto error_it = error.begin();
            error_it[i] = static_cast<time_type>(1e-8);
        }
        
        std::cout << "DEBUG: dop853_step completed" << std::endl;
    }
};
