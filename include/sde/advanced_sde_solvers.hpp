#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>
#include <array>

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
 * @brief Factory functions for creating standard tableaux
 */
namespace tableaux {

template<typename T = double>
SRATableau<T> constructSRA1() {
    SRATableau<T> tab;
    tab.stages = 2;
    tab.order = T(1.5);
    
    // Drift tableau A0
    tab.A0 = {{T(0), T(0)}, 
              {T(0.75), T(0)}};
    tab.c0 = {T(0), T(0.75)};
    tab.alpha = {T(1.0/3.0), T(2.0/3.0)};
    
    // Diffusion tableau B0
    tab.B0 = {{T(0), T(0)},
              {T(1.5), T(0)}};
    tab.c1 = {T(1), T(0)};
    tab.beta1 = {T(1), T(0)};
    tab.beta2 = {T(-1), T(1)};
    
    return tab;
}

template<typename T = double>
SRATableau<T> constructSRA2() {
    SRATableau<T> tab;
    tab.stages = 2;
    tab.order = T(1.5);
    
    tab.A0 = {{T(0), T(0)}, 
              {T(0.75), T(0)}};
    tab.c0 = {T(0), T(0.75)};
    tab.alpha = {T(1.0/3.0), T(2.0/3.0)};
    
    tab.B0 = {{T(0), T(0)},
              {T(1.5), T(0)}};
    tab.c1 = {T(1.0/3.0), T(1)};
    tab.beta1 = {T(0), T(1)};
    tab.beta2 = {T(1.5), T(-1.5)};
    
    return tab;
}

template<typename T = double>
SRATableau<T> constructSOSRA() {
    // Stability-optimized SRA coefficients
    SRATableau<T> tab;
    tab.stages = 3;
    tab.order = T(1.5);
    
    // Optimized coefficients for stability
    tab.alpha = {T(0.2889874966892885), T(0.6859880440839937), T(0.025024459226717772)};
    
    tab.A0 = {{T(0), T(0), T(0)},
              {T(0.6923962376159507), T(0), T(0)},
              {T(0.9511849235504364), T(0.04881507644956362), T(0)}};
    tab.c0 = {T(0), T(0.6923962376159507), T(1)};
    
    tab.B0 = {{T(0), T(0), T(0)},
              {T(0.7686101171003622), T(0), T(0)},
              {T(0.43886792994934987), T(0.7490415909204886), T(0)}};
    tab.c1 = {T(0), T(0.041248171110700504), T(1)};
    
    tab.beta1 = {T(-16.792534242221663), T(17.514995785380226), T(0.27753845684143835)};
    tab.beta2 = {T(0.4237535769069274), T(0.6010381474428539), T(-1.0247917243497813)};
    
    return tab;
}

template<typename T = double>
SRITableau<T> constructSRIW1() {
    SRITableau<T> tab;
    tab.stages = 4;
    tab.order = T(1.5);
    
    tab.c0 = {T(0), T(1), T(0.5), T(0)};
    tab.c1 = {T(1.0/3.0), T(2.0/3.0), T(0), T(0)};
    
    // Drift coefficients A0
    tab.A0 = {{T(0), T(0), T(0), T(0)},
              {T(1), T(0), T(0), T(0)},
              {T(0.25), T(0.25), T(0), T(0)},
              {T(1.0/6.0), T(1.0/6.0), T(2.0/3.0), T(0)}};
    
    // Drift coefficients A1 
    tab.A1 = {{T(0), T(0), T(0), T(0)},
              {T(0), T(0), T(0), T(0)},
              {T(-5), T(3), T(0), T(0)},
              {T(-5), T(3), T(0.5), T(0)}};
    
    tab.alpha = {T(1.0/3.0), T(2.0/3.0), T(0), T(0)};
    
    // Diffusion coefficients B0
    tab.B0 = {{T(0), T(0), T(0), T(0)},
              {T(2), T(0), T(0), T(0)},
              {T(1), T(1), T(0), T(0)},
              {T(1.0/3.0), T(2.0/3.0), T(0), T(0)}};
    
    // Diffusion coefficients B1
    tab.B1 = {{T(0), T(0), T(0), T(0)},
              {T(0), T(0), T(0), T(0)},
              {T(-10), T(6), T(0), T(0)},
              {T(-10), T(6), T(1), T(0)}};
    
    tab.beta1 = {T(-1), T(4.0/3.0), T(2.0/3.0), T(0)};
    tab.beta2 = {T(-1), T(4.0/3.0), T(-1.0/3.0), T(0)};
    tab.beta3 = {T(2), T(-4.0/3.0), T(-2.0/3.0), T(0)};
    tab.beta4 = {T(-2), T(5.0/3.0), T(-2.0/3.0), T(1)};
    
    return tab;
}

template<typename T = double>
SRITableau<T> constructSOSRI() {
    // Stability-optimized SRI coefficients
    SRITableau<T> tab;
    tab.stages = 4;
    tab.order = T(1.5);
    
    // Optimized coefficients for enhanced stability
    tab.alpha = {T(0.1978227786259619), T(0.5255669992543889), T(0.27661022211964925), T(0)};
    
    tab.c0 = {T(0), T(0.2607835959202928), T(0.42199503108465265), T(1)};
    tab.c1 = {T(0), T(0.05991061344301315), T(0.3254461950380998), T(1)};
    
    // Optimized drift coefficients
    tab.A0 = {{T(0), T(0), T(0), T(0)},
              {T(0.2607835959202928), T(0), T(0), T(0)},
              {T(0.1842055156142715), T(0.2377895154703812), T(0), T(0)},
              {T(0.1694266494680467), T(0.15493207673976627), T(0.675640273791867), T(0)}};
    
    // Additional stability terms for A1
    tab.A1 = {{T(0), T(0), T(0), T(0)},
              {T(0), T(0), T(0), T(0)},
              {T(-0.8473228402389086), T(0.6019001037086358), T(0), T(0)},
              {T(-0.8051905634073717), T(0.5731769288842242), T(0.232013634523147), T(0)}};
    
    // Diffusion coefficients optimized for stability
    tab.B0 = {{T(0), T(0), T(0), T(0)},
              {T(0.1197648236344262), T(0), T(0), T(0)},
              {T(0.08139976719005145), T(0.24457513264396382), T(0), T(0)},
              {T(0.08139976719005145), T(0.24457513264396382), T(0), T(0)}};
    
    tab.B1 = {{T(0), T(0), T(0), T(0)},
              {T(0), T(0), T(0), T(0)},
              {T(-0.16279953438010288), T(0.4891502652879276), T(0), T(0)},
              {T(-0.16279953438010288), T(0.4891502652879276), T(0), T(0)}};
    
    // Optimized beta coefficients
    tab.beta1 = {T(0.8936346481982765), T(0.5459984306999252), T(-0.4396330788982017), T(0)};
    tab.beta2 = {T(-1.7872692963965531), T(-1.0919968613998504), T(0.8792661577964034), T(0)};
    tab.beta3 = {T(1.0729697095491348), T(0.6551981168399102), T(-0.5275597146778420), T(0)};
    tab.beta4 = {T(-0.08936346481982765), T(-0.05459984306999252), T(0.04396330788982017), T(1)};
    
    return tab;
}

} // namespace tableaux

/**
 * @brief SRA (Stochastic Runge-Kutta for Additive noise) integrator
 * 
 * Implements the SRA family of methods for additive noise SDEs:
 * dX = f(t, X) dt + g(t) dW
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
                          tableau_type tableau = tableaux::constructSRA1<value_type>())
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
template<system_state StateType, can_be_time TimeType>
class SRIIntegrator : public AbstractSDEIntegrator<StateType, TimeType> {
public:
    using base_type = AbstractSDEIntegrator<StateType, TimeType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using tableau_type = SRITableau<value_type>;
    
    explicit SRIIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                          std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr,
                          tableau_type tableau = tableaux::constructSRIW1<value_type>())
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
};

/**
 * @brief Convenience aliases for specific methods
 */
template<system_state StateType, can_be_time TimeType>
using SRA1Integrator = SRAIntegrator<StateType, TimeType>;

template<system_state StateType, can_be_time TimeType>
using SRA2Integrator = SRAIntegrator<StateType, TimeType>;

template<system_state StateType, can_be_time TimeType>
using SOSRAIntegrator = SRAIntegrator<StateType, TimeType>;

template<system_state StateType, can_be_time TimeType>
using SRIW1Integrator = SRIIntegrator<StateType, TimeType>;

template<system_state StateType, can_be_time TimeType>
using SOSRIIntegrator = SRIIntegrator<StateType, TimeType>;

/**
 * @brief Factory functions for creating specific SDE integrators
 */
namespace factory {

template<system_state StateType, can_be_time TimeType>
auto make_sra1_integrator(std::shared_ptr<SDEProblem<StateType, TimeType>> problem,
                         std::shared_ptr<WienerProcess<StateType, TimeType>> wiener = nullptr) {
    return std::make_unique<SRAIntegrator<StateType, TimeType>>(
        problem, wiener, tableaux::constructSRA1<typename StateType::value_type>());
}

template<system_state StateType, can_be_time TimeType>
auto make_sra2_integrator(std::shared_ptr<SDEProblem<StateType, TimeType>> problem,
                         std::shared_ptr<WienerProcess<StateType, TimeType>> wiener = nullptr) {
    return std::make_unique<SRAIntegrator<StateType, TimeType>>(
        problem, wiener, tableaux::constructSRA2<typename StateType::value_type>());
}

template<system_state StateType, can_be_time TimeType>
auto make_sosra_integrator(std::shared_ptr<SDEProblem<StateType, TimeType>> problem,
                          std::shared_ptr<WienerProcess<StateType, TimeType>> wiener = nullptr) {
    return std::make_unique<SRAIntegrator<StateType, TimeType>>(
        problem, wiener, tableaux::constructSOSRA<typename StateType::value_type>());
}

template<system_state StateType, can_be_time TimeType>
auto make_sriw1_integrator(std::shared_ptr<SDEProblem<StateType, TimeType>> problem,
                          std::shared_ptr<WienerProcess<StateType, TimeType>> wiener = nullptr) {
    return std::make_unique<SRIIntegrator<StateType, TimeType>>(
        problem, wiener, tableaux::constructSRIW1<typename StateType::value_type>());
}

template<system_state StateType, can_be_time TimeType>
auto make_sosri_integrator(std::shared_ptr<SDEProblem<StateType, TimeType>> problem,
                          std::shared_ptr<WienerProcess<StateType, TimeType>> wiener = nullptr) {
    return std::make_unique<SRIIntegrator<StateType, TimeType>>(
        problem, wiener, tableaux::constructSOSRI<typename StateType::value_type>());
}

} // namespace factory

} // namespace diffeq::sde
