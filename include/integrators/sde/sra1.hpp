#pragma once

#include <integrators/sde/sra.hpp>

namespace diffeq::sde {

/**
 * @brief SRA1 integrator variant
 * 
 * SRA integrator configured with SRA1 tableau coefficients.
 * Optimized for additive noise SDEs with strong order 1.5.
 */
template<system_state StateType, can_be_time TimeType>
class SRA1Integrator : public SRAIntegrator<StateType, TimeType> {
public:
    using base_type = SRAIntegrator<StateType, TimeType>;
    
    explicit SRA1Integrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                           std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener, create_sra1_tableau()) {}
    
    std::string name() const override {
        return "SRA1 (Strong Order 1.5 for Additive Noise)";
    }

private:
    static typename base_type::tableau_type create_sra1_tableau() {
        typename base_type::tableau_type tableau;
        tableau.stages = 2;
        tableau.order = static_cast<typename base_type::value_type>(1.5);
        
        // SRA1 drift coefficients
        tableau.A0 = {{0, 0}, {1, 0}};
        tableau.c0 = {0, 1};
        tableau.alpha = {static_cast<typename base_type::value_type>(0.5), 
                        static_cast<typename base_type::value_type>(0.5)};
        
        // SRA1 diffusion coefficients
        tableau.B0 = {{0, 0}, {1, 0}};
        tableau.c1 = {0, 1};
        tableau.beta1 = {static_cast<typename base_type::value_type>(0.5), 
                        static_cast<typename base_type::value_type>(0.5)};
        tableau.beta2 = {0, 1};
        
        return tableau;
    }
};

} // namespace diffeq::sde
