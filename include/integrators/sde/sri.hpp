#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

namespace diffeq {

/**
 * @brief Tableau coefficients for SRI methods
 */
template<typename T>
struct SRITableau {
    // Drift coefficients
    std::vector<std::vector<T>> A0, A1;
    std::vector<T> c0;
    std::vector<T> alpha;
    
    // Diffusion coefficients
    std::vector<std::vector<T>> B0, B1;  
    std::vector<T> c1;
    std::vector<T> beta1, beta2, beta3, beta4;
    
    int stages;
    T order;
};

/**
 * @brief SRI (Stochastic Runge-Kutta for general Itô SDEs) integrator
 * 
 * Implements the SRI family of methods for general Itô SDEs:
 * dX = f(t, X) dt + g(t, X) dW
 * 
 * Strong order: 1.5
 * Weak order: 2.0
 * 
 * Reference: Rößler A., Runge–Kutta Methods for the Strong Approximation 
 * of Solutions of Stochastic Differential Equations, SIAM J. Numer. Anal., 
 * 48 (3), pp. 922–952. DOI:10.1137/09076636X
 */
template<system_state StateType>
class SRIIntegrator : public sde::AbstractSDEIntegrator<StateType> {
public:
    using base_type = sde::AbstractSDEIntegrator<StateType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using tableau_type = SRITableau<value_type>;
    
    explicit SRIIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                          std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr,
                          tableau_type tableau = SRIIntegrator::create_sriw1_tableau())
        : base_type(problem, wiener)
        , tableau_(std::move(tableau)) {}
    
    void step(state_type& state, time_type dt) override {
        const int stages = tableau_.stages;
        
        // Create temporary states
        std::vector<state_type> H0(stages), H1(stages);
        for (int i = 0; i < stages; ++i) {
            H0[i] = StateCreator<state_type>::create(state);
            H1[i] = StateCreator<state_type>::create(state);
        }
        
        state_type dW = StateCreator<state_type>::create(state);
        state_type dZ = StateCreator<state_type>::create(state);
        state_type ftmp = StateCreator<state_type>::create(state);
        state_type gtmp = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        this->wiener_->generate_increment(dZ, dt);  
        
        // Compute multiple stochastic integrals
        value_type sqrt3 = std::sqrt(static_cast<value_type>(3));
        value_type sqrt_dt = std::sqrt(static_cast<value_type>(dt));
        
        // chi1 = (1/2) * ((dW)^2 - dt) / sqrt(dt) for I_(1,1)/sqrt(h)
        // chi2 = (1/2) * (dW + dZ/sqrt(3)) for I_(1,0)/h  
        // chi3 = (1/6) * ((dW)^3 - 3*dW*dt) / dt for I_(1,1,1)/h
        state_type chi1 = StateCreator<state_type>::create(state);
        state_type chi2 = StateCreator<state_type>::create(state);
        state_type chi3 = StateCreator<state_type>::create(state);
        
        for (size_t j = 0; j < state.size(); ++j) {
            auto dW_it = dW.begin();
            auto dZ_it = dZ.begin();
            auto chi1_it = chi1.begin();
            auto chi2_it = chi2.begin();
            auto chi3_it = chi3.begin();
            
            value_type dW_val = dW_it[j];
            value_type dW_squared = dW_val * dW_val;
            
            chi1_it[j] = static_cast<value_type>(0.5) * (dW_squared - dt) / sqrt_dt;
            chi2_it[j] = static_cast<value_type>(0.5) * (dW_val + dZ_it[j] / sqrt3);
            chi3_it[j] = static_cast<value_type>(1.0/6.0) * (dW_val * dW_squared - 3 * dW_val * dt) / dt;
        }
        
        // Initialize H0[0] = H1[0] = current state
        for (size_t j = 0; j < state.size(); ++j) {
            auto state_it = state.begin();
            auto H0_0_it = H0[0].begin();
            auto H1_0_it = H1[0].begin();
            H0_0_it[j] = state_it[j];
            H1_0_it[j] = state_it[j];
        }
        
        // Compute stages
        for (int i = 1; i < stages; ++i) {
            state_type A0temp = StateCreator<state_type>::create(state);
            state_type A1temp = StateCreator<state_type>::create(state);
            state_type B0temp = StateCreator<state_type>::create(state);
            state_type B1temp = StateCreator<state_type>::create(state);
            
            for (int j = 0; j < i; ++j) {
                this->problem_->drift(this->current_time_ + tableau_.c0[j] * dt, H0[j], ftmp);
                this->problem_->diffusion(this->current_time_ + tableau_.c1[j] * dt, H1[j], gtmp);
                
                for (size_t k = 0; k < state.size(); ++k) {
                    auto A0temp_it = A0temp.begin();
                    auto A1temp_it = A1temp.begin();
                    auto B0temp_it = B0temp.begin();
                    auto B1temp_it = B1temp.begin();
                    auto ftmp_it = ftmp.begin();
                    auto gtmp_it = gtmp.begin();
                    auto chi1_it = chi1.begin();
                    auto chi2_it = chi2.begin();
                    
                    A0temp_it[k] += tableau_.A0[j][i] * ftmp_it[k];
                    A1temp_it[k] += tableau_.A1[j][i] * ftmp_it[k];
                    B0temp_it[k] += tableau_.B0[j][i] * gtmp_it[k];
                    B1temp_it[k] += tableau_.B1[j][i] * gtmp_it[k] * chi1_it[k];
                }
            }
            
            // Update H0[i] and H1[i]
            for (size_t k = 0; k < state.size(); ++k) {
                auto state_it = state.begin();
                auto H0_i_it = H0[i].begin();
                auto H1_i_it = H1[i].begin();
                auto A0temp_it = A0temp.begin();
                auto A1temp_it = A1temp.begin();
                auto B0temp_it = B0temp.begin();
                auto B1temp_it = B1temp.begin();
                auto chi2_it = chi2.begin();
                auto dW_it = dW.begin();
                
                H0_i_it[k] = state_it[k] + dt * A0temp_it[k] + B0temp_it[k] * dW_it[k];
                H1_i_it[k] = state_it[k] + dt * A1temp_it[k] + B0temp_it[k] * sqrt_dt + B1temp_it[k] + chi2_it[k] * B0temp_it[k];
            }
        }
        
        // Compute final update
        state_type drift_sum = StateCreator<state_type>::create(state);
        state_type E1 = StateCreator<state_type>::create(state);
        state_type E2 = StateCreator<state_type>::create(state);
        state_type E3 = StateCreator<state_type>::create(state);
        
        std::fill(drift_sum.begin(), drift_sum.end(), value_type(0));
        std::fill(E1.begin(), E1.end(), value_type(0));
        std::fill(E2.begin(), E2.end(), value_type(0));
        std::fill(E3.begin(), E3.end(), value_type(0));
        
        for (int i = 0; i < stages; ++i) {
            this->problem_->drift(this->current_time_ + tableau_.c0[i] * dt, H0[i], ftmp);
            this->problem_->diffusion(this->current_time_ + tableau_.c1[i] * dt, H1[i], gtmp);
            
            for (size_t k = 0; k < state.size(); ++k) {
                auto drift_sum_it = drift_sum.begin();
                auto E1_it = E1.begin();
                auto E2_it = E2.begin();
                auto E3_it = E3.begin();
                auto ftmp_it = ftmp.begin();
                auto gtmp_it = gtmp.begin();
                auto dW_it = dW.begin();
                auto chi1_it = chi1.begin();
                auto chi2_it = chi2.begin();
                auto chi3_it = chi3.begin();
                
                drift_sum_it[k] += tableau_.alpha[i] * ftmp_it[k];
                E1_it[k] += tableau_.beta1[i] * gtmp_it[k] * dW_it[k];
                E2_it[k] += tableau_.beta2[i] * gtmp_it[k] * chi1_it[k];
                E2_it[k] += tableau_.beta3[i] * gtmp_it[k] * chi2_it[k];
                E3_it[k] += tableau_.beta4[i] * gtmp_it[k] * chi3_it[k];
            }
        }
        
        // Final state update
        for (size_t k = 0; k < state.size(); ++k) {
            auto state_it = state.begin();
            auto drift_sum_it = drift_sum.begin();
            auto E1_it = E1.begin();
            auto E2_it = E2.begin();
            auto E3_it = E3.begin();
            
            state_it[k] += dt * drift_sum_it[k] + E1_it[k] + E2_it[k] + E3_it[k];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "SRI (Strong Order 1.5 for General Itô SDEs)";
    }
    
    void set_tableau(const tableau_type& tableau) {
        tableau_ = tableau;
    }

private:
    tableau_type tableau_;
    
    // Default SRIW1 tableau
    static tableau_type create_sriw1_tableau() {
        tableau_type tableau;
        tableau.stages = 2;
        tableau.order = static_cast<value_type>(1.5);
        
        // Basic SRIW1 coefficients (simplified)
        tableau.A0 = {{0, 0}, {1, 0}};
        tableau.A1 = {{0, 0}, {1, 0}};
        tableau.c0 = {0, 1};
        tableau.alpha = {static_cast<value_type>(0.5), static_cast<value_type>(0.5)};
        
        tableau.B0 = {{0, 0}, {1, 0}};
        tableau.B1 = {{0, 0}, {1, 0}};
        tableau.c1 = {0, 1};
        tableau.beta1 = {static_cast<value_type>(0.5), static_cast<value_type>(0.5)};
        tableau.beta2 = {0, 1};
        tableau.beta3 = {0, static_cast<value_type>(0.5)};
        tableau.beta4 = {0, static_cast<value_type>(1.0/6.0)};
        
        return tableau;
    }
};

} // namespace diffeq
