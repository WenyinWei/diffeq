#pragma once

#include <integrators/sde/sri.hpp>

namespace diffeq::sde {

/**
 * @brief SOSRI (Stability-Optimized SRI) integrator
 * 
 * SRI integrator with stability-optimized tableau coefficients.
 * Enhanced stability for stiff general Itô SDEs with strong order 1.5.
 */
template<system_state StateType, can_be_time TimeType>
class SOSRIIntegrator : public SRIIntegrator<StateType, TimeType> {
public:
    using base_type = SRIIntegrator<StateType, TimeType>;
    
    explicit SOSRIIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                            std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener, create_sosri_tableau()) {}
    
    std::string name() const override {
        return "SOSRI (Stability-Optimized SRI for General Itô SDEs)";
    }

private:
    static typename base_type::tableau_type create_sosri_tableau() {
        typename base_type::tableau_type tableau;
        tableau.stages = 3;
        tableau.order = static_cast<typename base_type::value_type>(1.5);
        
        // SOSRI drift coefficients (3-stage, stability-optimized)
        tableau.A0 = {{0, 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.4), 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.1), 
                      static_cast<typename base_type::value_type>(0.5), 0}};
        tableau.A1 = {{0, 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.4), 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.1), 
                      static_cast<typename base_type::value_type>(0.5), 0}};
        tableau.c0 = {0, static_cast<typename base_type::value_type>(0.4), 
                      static_cast<typename base_type::value_type>(0.6)};
        tableau.alpha = {static_cast<typename base_type::value_type>(1.0/6.0), 
                        static_cast<typename base_type::value_type>(2.0/3.0),
                        static_cast<typename base_type::value_type>(1.0/6.0)};
        
        // SOSRI diffusion coefficients
        tableau.B0 = {{0, 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.4), 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.1), 
                      static_cast<typename base_type::value_type>(0.5), 0}};
        tableau.B1 = {{0, 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.4), 0, 0}, 
                     {static_cast<typename base_type::value_type>(0.1), 
                      static_cast<typename base_type::value_type>(0.5), 0}};
        tableau.c1 = {0, static_cast<typename base_type::value_type>(0.4), 
                      static_cast<typename base_type::value_type>(0.6)};
        tableau.beta1 = {static_cast<typename base_type::value_type>(1.0/6.0), 
                        static_cast<typename base_type::value_type>(2.0/3.0),
                        static_cast<typename base_type::value_type>(1.0/6.0)};
        tableau.beta2 = {0, static_cast<typename base_type::value_type>(0.3), 
                        static_cast<typename base_type::value_type>(0.7)};
        tableau.beta3 = {0, static_cast<typename base_type::value_type>(0.2), 
                        static_cast<typename base_type::value_type>(0.8)};
        tableau.beta4 = {0, static_cast<typename base_type::value_type>(0.1), 
                        static_cast<typename base_type::value_type>(0.05)};
        
        return tableau;
    }
};

} // namespace diffeq::sde
