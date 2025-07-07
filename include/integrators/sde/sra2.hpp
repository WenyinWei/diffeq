#pragma once

#include <integrators/sde/sra.hpp>
#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq {

/**
 * @brief SRA2 integrator variant
 * 
 * SRA integrator configured with SRA2 tableau coefficients.
 * Alternative SRA tableau for additive noise SDEs with strong order 1.5.
 */
template<system_state StateType>
class SRA2Integrator : public SRAIntegrator<StateType> {
public:
    using base_type = SRAIntegrator<StateType>;
    
    explicit SRA2Integrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                           std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener, create_sra2_tableau()) {}
    
    std::string name() const override {
        return "SRA2 (Strong Order 1.5 for Additive Noise)";
    }

private:
    static typename base_type::tableau_type create_sra2_tableau() {
        typename base_type::tableau_type tableau;
        tableau.stages = 3;
        tableau.order = static_cast<typename base_type::value_type>(1.5);
        
        // SRA2 drift coefficients (3-stage method)
        tableau.A0 = {{0, 0, 0}, {static_cast<typename base_type::value_type>(0.5), 0, 0}, 
                     {0, static_cast<typename base_type::value_type>(0.75), 0}};
        tableau.c0 = {0, static_cast<typename base_type::value_type>(0.5), 
                      static_cast<typename base_type::value_type>(0.75)};
        tableau.alpha = {static_cast<typename base_type::value_type>(2.0/9.0), 
                        static_cast<typename base_type::value_type>(1.0/3.0),
                        static_cast<typename base_type::value_type>(4.0/9.0)};
        
        // SRA2 diffusion coefficients  
        tableau.B0 = {{0, 0, 0}, {static_cast<typename base_type::value_type>(0.5), 0, 0}, 
                     {0, static_cast<typename base_type::value_type>(0.75), 0}};
        tableau.c1 = {0, static_cast<typename base_type::value_type>(0.5), 
                      static_cast<typename base_type::value_type>(0.75)};
        tableau.beta1 = {static_cast<typename base_type::value_type>(2.0/9.0), 
                        static_cast<typename base_type::value_type>(1.0/3.0),
                        static_cast<typename base_type::value_type>(4.0/9.0)};
        tableau.beta2 = {0, static_cast<typename base_type::value_type>(0.5), 
                        static_cast<typename base_type::value_type>(0.5)};
        
        return tableau;
    }
};

} // namespace diffeq
