#pragma once

#include <sde/sde_base.hpp>
#include <core/state_creator.hpp>
#include <cmath>

namespace diffeq {

/**
 * @brief SOSRA (Stability-Optimized SRA) integrator
 * 
 * SRA integrator with stability-optimized tableau coefficients.
 * Enhanced stability for stiff additive noise SDEs with strong order 1.5.
 */
template<system_state StateType>
class SOSRAIntegrator : public AbstractSDEIntegrator<StateType> {
public:
    using base_type = AbstractSDEIntegrator<StateType>;
    
    explicit SOSRAIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                            std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener, create_sosra_tableau()) {}
    
    std::string name() const override {
        return "SOSRA (Stability-Optimized SRA for Additive Noise)";
    }

private:
    static typename base_type::tableau_type create_sosra_tableau() {
        typename base_type::tableau_type tableau;
        tableau.stages = 2;
        tableau.order = static_cast<typename base_type::value_type>(1.5);
        
        // SOSRA drift coefficients (stability-optimized)
        tableau.A0 = {{0, 0}, {static_cast<typename base_type::value_type>(0.6), 0}};
        tableau.c0 = {0, static_cast<typename base_type::value_type>(0.6)};
        tableau.alpha = {static_cast<typename base_type::value_type>(0.4), 
                        static_cast<typename base_type::value_type>(0.6)};
        
        // SOSRA diffusion coefficients
        tableau.B0 = {{0, 0}, {static_cast<typename base_type::value_type>(0.6), 0}};
        tableau.c1 = {0, static_cast<typename base_type::value_type>(0.6)};
        tableau.beta1 = {static_cast<typename base_type::value_type>(0.4), 
                        static_cast<typename base_type::value_type>(0.6)};
        tableau.beta2 = {static_cast<typename base_type::value_type>(-0.1), 
                        static_cast<typename base_type::value_type>(1.1)};
        
        return tableau;
    }
};

} // namespace diffeq
