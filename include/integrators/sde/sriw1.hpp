#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq {

/**
 * @brief SRIW1 integrator variant
 * 
 * SRI integrator configured with SRIW1 tableau coefficients.
 * Weak order 2.0 method for general Itô SDEs with strong order 1.5.
 */
template<system_state StateType>
class SRIW1Integrator : public sde::AbstractSDEIntegrator<StateType> {
public:
    using base_type = sde::AbstractSDEIntegrator<StateType>;
    
    explicit SRIW1Integrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                            std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener, create_sriw1_tableau()) {}
    
    std::string name() const override {
        return "SRIW1 (Strong Order 1.5, Weak Order 2.0 for General Itô SDEs)";
    }

private:
    static typename base_type::tableau_type create_sriw1_tableau() {
        typename base_type::tableau_type tableau;
        tableau.stages = 2;
        tableau.order = static_cast<typename base_type::value_type>(1.5);
        
        // SRIW1 drift coefficients
        tableau.A0 = {{0, 0}, {1, 0}};
        tableau.A1 = {{0, 0}, {1, 0}};
        tableau.c0 = {0, 1};
        tableau.alpha = {static_cast<typename base_type::value_type>(0.5), 
                        static_cast<typename base_type::value_type>(0.5)};
        
        // SRIW1 diffusion coefficients
        tableau.B0 = {{0, 0}, {1, 0}};
        tableau.B1 = {{0, 0}, {1, 0}};
        tableau.c1 = {0, 1};
        tableau.beta1 = {static_cast<typename base_type::value_type>(0.5), 
                        static_cast<typename base_type::value_type>(0.5)};
        tableau.beta2 = {0, 1};
        tableau.beta3 = {0, static_cast<typename base_type::value_type>(0.5)};
        tableau.beta4 = {0, static_cast<typename base_type::value_type>(1.0/6.0)};
        
        return tableau;
    }
};

} // namespace diffeq
