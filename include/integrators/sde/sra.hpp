#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <vector>

namespace diffeq::sde {

/**
 * @brief Tableau coefficients for SRA methods
 */
template<typename T>
struct SRATableau {
    // Drift coefficients
    std::vector<std::vector<T>> A0;
    std::vector<T> c0;
    std::vector<T> alpha;
    
    // Diffusion coefficients  
    std::vector<std::vector<T>> B0;
    std::vector<T> c1;
    std::vector<T> beta1;
    std::vector<T> beta2;
    
    int stages;
    T order;
};

/**
 * @brief SRA (Stochastic Runge-Kutta for additive noise SDEs) integrator
 * 
 * Implements the SRA family of methods for additive noise SDEs:
 * dX = f(t, X) dt + g(t, X) dW
 * 
 * Strong order: 1.5
 * Weak order: 2.0
 * 
 * Reference: Rößler A., Runge–Kutta Methods for the Strong Approximation 
 * of Solutions of Stochastic Differential Equations, SIAM J. Numer. Anal., 
 * 48 (3), pp. 922–952. DOI:10.1137/09076636X
 */
template<system_state StateType, can_be_time TimeType>
class SRAIntegrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using tableau_type = SRATableau<value_type>;
    
    explicit SRAIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                          std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr,
                          tableau_type tableau = SRAIntegrator::create_sra1_tableau())
        : base_type(problem, wiener)
        , tableau_(std::move(tableau)) {}
    
    void step(state_type& state, time_type dt) override {
        const int stages = tableau_.stages;
        
        // Create temporary states
        std::vector<state_type> H0(stages);
        for (int i = 0; i < stages; ++i) {
            H0[i] = StateCreator<state_type>::create(state);
        }
        
        state_type dW = StateCreator<state_type>::create(state);
        state_type dZ = StateCreator<state_type>::create(state);  // For chi2 computation
        state_type ftmp = StateCreator<state_type>::create(state);
        state_type gtmp = StateCreator<state_type>::create(state);
        state_type atemp = StateCreator<state_type>::create(state);
        state_type btemp = StateCreator<state_type>::create(state);
        state_type E2 = StateCreator<state_type>::create(state);
        
        // Generate Wiener increments  
        this->wiener_->generate_increment(dW, dt);
        this->wiener_->generate_increment(dZ, dt);  // Independent for chi2
        
        // Compute chi2 = (1/2)*(dW + dZ/sqrt(3)) for I_(1,0)/h
        value_type sqrt3 = std::sqrt(static_cast<value_type>(3));
        state_type chi2 = StateCreator<state_type>::create(state);
        for (size_t j = 0; j < chi2.size(); ++j) {
            auto chi2_it = chi2.begin();
            auto dW_it = dW.begin();
            auto dZ_it = dZ.begin();
            chi2_it[j] = static_cast<value_type>(0.5) * (dW_it[j] + dZ_it[j] / sqrt3);
        }
        
        // Initialize H0[0] = current state
        for (size_t j = 0; j < state.size(); ++j) {
            auto state_it = state.begin();
            auto H0_0_it = H0[0].begin();
            H0_0_it[j] = state_it[j];
        }
        
        // Compute stages
        for (int i = 1; i < stages; ++i) {
            // Compute A0temp and B0temp
            state_type A0temp = StateCreator<state_type>::create(state);
            state_type B0temp = StateCreator<state_type>::create(state);
            
            for (int j = 0; j < i; ++j) {
                this->problem_->drift(this->current_time_ + tableau_.c0[j] * dt, H0[j], ftmp);
                this->problem_->diffusion(this->current_time_ + tableau_.c1[j] * dt, H0[j], gtmp);
                
                for (size_t k = 0; k < state.size(); ++k) {
                    auto A0temp_it = A0temp.begin();
                    auto B0temp_it = B0temp.begin();
                    auto ftmp_it = ftmp.begin();
                    auto gtmp_it = gtmp.begin();
                    auto chi2_it = chi2.begin();
                    
                    A0temp_it[k] += tableau_.A0[j][i] * ftmp_it[k];
                    B0temp_it[k] += tableau_.B0[j][i] * gtmp_it[k] * chi2_it[k];
                }
            }
            
            // H0[i] = state + dt*A0temp + B0temp
            for (size_t k = 0; k < state.size(); ++k) {
                auto state_it = state.begin();
                auto H0_i_it = H0[i].begin();
                auto A0temp_it = A0temp.begin();
                auto B0temp_it = B0temp.begin();
                
                H0_i_it[k] = state_it[k] + dt * A0temp_it[k] + B0temp_it[k];
            }
        }
        
        // Compute final update terms
        std::fill(atemp.begin(), atemp.end(), value_type(0));
        std::fill(btemp.begin(), btemp.end(), value_type(0));
        std::fill(E2.begin(), E2.end(), value_type(0));
        
        for (int i = 0; i < stages; ++i) {
            this->problem_->drift(this->current_time_ + tableau_.c0[i] * dt, H0[i], ftmp);
            this->problem_->diffusion(this->current_time_ + tableau_.c1[i] * dt, H0[i], gtmp);
            
            for (size_t k = 0; k < state.size(); ++k) {
                auto atemp_it = atemp.begin();
                auto btemp_it = btemp.begin();
                auto E2_it = E2.begin();
                auto ftmp_it = ftmp.begin();
                auto gtmp_it = gtmp.begin();
                auto dW_it = dW.begin();
                auto chi2_it = chi2.begin();
                
                atemp_it[k] += tableau_.alpha[i] * ftmp_it[k];
                btemp_it[k] += tableau_.beta1[i] * gtmp_it[k] * dW_it[k];
                E2_it[k] += tableau_.beta2[i] * gtmp_it[k] * chi2_it[k];
            }
        }
        
        // Final state update: u = uprev + dt*atemp + btemp + E2
        for (size_t k = 0; k < state.size(); ++k) {
            auto state_it = state.begin();
            auto atemp_it = atemp.begin();
            auto btemp_it = btemp.begin();
            auto E2_it = E2.begin();
            
            state_it[k] += dt * atemp_it[k] + btemp_it[k] + E2_it[k];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "SRA (Strong Order 1.5 for Additive Noise)";
    }
    
    void set_tableau(const tableau_type& tableau) {
        tableau_ = tableau;
    }

private:
    tableau_type tableau_;
    
    // Default SRA1 tableau
    static tableau_type create_sra1_tableau() {
        tableau_type tableau;
        tableau.stages = 2;
        tableau.order = static_cast<value_type>(1.5);
        
        // Drift coefficients
        tableau.A0 = {{0, 0}, {1, 0}};
        tableau.c0 = {0, 1};
        tableau.alpha = {static_cast<value_type>(0.5), static_cast<value_type>(0.5)};
        
        // Diffusion coefficients
        tableau.B0 = {{0, 0}, {1, 0}};
        tableau.c1 = {0, 1};
        tableau.beta1 = {static_cast<value_type>(0.5), static_cast<value_type>(0.5)};
        tableau.beta2 = {0, 1};
        
        return tableau;
    }
};

} // namespace diffeq::sde
