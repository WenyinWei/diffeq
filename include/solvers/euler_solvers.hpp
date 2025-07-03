#pragma once
#include <functional>
#include <concepts>
#include <iterator>
#include <type_traits>
#include <vector>
#include <array>
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>

// Helper trait to create state objects (shared with RK4)
template<system_state S>
struct StateCreator {
    static S create(const S& template_state) {
        if constexpr (requires { S(template_state.size()); }) {
            // For containers like std::vector that can be constructed with size
            return S(template_state.size());
        } else {
            // For fixed-size containers like std::array, copy and zero out
            S result = template_state;
            for (auto& val : result) {
                val = typename S::value_type{};
            }
            return result;
        }
    }
};

// Simple Euler integrator: y_{n+1} = y_n + h * f(t_n, y_n)
template<system_state S, can_be_time T = double>
class EulerIntegrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit EulerIntegrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary state for derivative
        state_type dydt = StateCreator<state_type>::create(state);
        
        // Compute derivative: dydt = f(t, y)
        this->sys_(this->current_time_, state, dydt);
        
        // Update state: y_new = y + dt * dydt
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto dydt_it = dydt.begin();
            
            state_it[i] = state_it[i] + dt * dydt_it[i];
        }
        
        this->advance_time(dt);
    }
};

// Improved Euler (Heun's method): y_{n+1} = y_n + h/2 * (k1 + k2)
// where k1 = f(t_n, y_n) and k2 = f(t_n + h, y_n + h*k1)
template<system_state S, can_be_time T = double>
class ImprovedEulerIntegrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit ImprovedEulerIntegrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary states
        state_type k1 = StateCreator<state_type>::create(state);
        state_type k2 = StateCreator<state_type>::create(state);
        state_type temp_state = StateCreator<state_type>::create(state);
        
        // k1 = f(t, y)
        this->sys_(this->current_time_, state, k1);
        
        // temp_state = y + dt * k1
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp_state.begin();
            
            temp_it[i] = state_it[i] + dt * k1_it[i];
        }
        
        // k2 = f(t + dt, temp_state)
        this->sys_(this->current_time_ + dt, temp_state, k2);
        
        // y_new = y + dt/2 * (k1 + k2)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            
            state_it[i] = state_it[i] + dt * (k1_it[i] + k2_it[i]) / static_cast<time_type>(2);
        }
        
        this->advance_time(dt);
    }
};
