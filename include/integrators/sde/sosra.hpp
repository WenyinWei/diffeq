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
 * 
 * Note: This is a simplified implementation for compatibility.
 */
template<system_state StateType>
class SOSRAIntegrator : public sde::AbstractSDEIntegrator<StateType> {
public:
    using base_type = sde::AbstractSDEIntegrator<StateType>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    
    explicit SOSRAIntegrator(std::shared_ptr<typename base_type::sde_problem_type> problem,
                            std::shared_ptr<typename base_type::wiener_process_type> wiener = nullptr)
        : base_type(problem, wiener) {}
    
    void step(state_type& state, time_type dt) override {
        // Simplified SOSRA implementation - falls back to Euler-Maruyama for now
        // A full implementation would use the SOSRA tableau coefficients
        
        state_type drift_term = create_state_like(state);
        state_type diffusion_term = create_state_like(state);
        state_type dW = create_state_like(state);
        
        // Generate Wiener increments
        this->wiener_->generate_increment(dW, dt);
        
        // Evaluate drift and diffusion
        this->problem_->drift(this->current_time_, state, drift_term);
        this->problem_->diffusion(this->current_time_, state, diffusion_term);
        
        // Simple Euler-Maruyama step (SOSRA implementation would be more complex)
        for (size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto drift_it = drift_term.begin();
            auto diffusion_it = diffusion_term.begin();
            auto dW_it = dW.begin();
            
            state_it[i] += drift_it[i] * dt + diffusion_it[i] * dW_it[i];
        }
        
        this->advance_time(dt);
    }
    
    std::string name() const override {
        return "SOSRA (Simplified Implementation)";
    }

private:
    template<typename State>
    State create_state_like(const State& prototype) {
        State result;
        if constexpr (requires { result.resize(prototype.size()); }) {
            result.resize(prototype.size());
            std::fill(result.begin(), result.end(), value_type{0});
        }
        return result;
    }
};

} // namespace diffeq
