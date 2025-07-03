#pragma once
#include <functional>
#include <concepts>
#include <iterator>
#include <type_traits>
#include <vector>
#include <array>
#include <core/concepts.hpp>
#include <core/integrator_concept.hpp>

// Helper trait to create state objects
template<State S>
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

template<State S, TimeType T = double>
class RK4Integrator : public AbstractIntegrator<S, T> {
public:
    using base_type = AbstractIntegrator<S, T>;
    using state_type = typename base_type::state_type;
    using time_type = typename base_type::time_type;
    using value_type = typename base_type::value_type;
    using system_function = typename base_type::system_function;

    explicit RK4Integrator(system_function sys)
        : base_type(std::move(sys)) {}

    void step(state_type& state, time_type dt) override {
        // Create temporary states for RK4 calculations
        state_type k1 = StateCreator<state_type>::create(state);
        state_type k2 = StateCreator<state_type>::create(state);
        state_type k3 = StateCreator<state_type>::create(state);
        state_type k4 = StateCreator<state_type>::create(state);
        state_type temp_state = StateCreator<state_type>::create(state);

        // k1 = f(t, y)
        this->sys_(this->current_time_, state, k1);
        
        // k2 = f(t + dt/2, y + dt*k1/2)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto temp_it = temp_state.begin();
            
            temp_it[i] = state_it[i] + dt * k1_it[i] / static_cast<time_type>(2);
        }
        this->sys_(this->current_time_ + dt / static_cast<time_type>(2), temp_state, k2);
        
        // k3 = f(t + dt/2, y + dt*k2/2)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k2_it = k2.begin();
            auto temp_it = temp_state.begin();
            
            temp_it[i] = state_it[i] + dt * k2_it[i] / static_cast<time_type>(2);
        }
        this->sys_(this->current_time_ + dt / static_cast<time_type>(2), temp_state, k3);
        
        // k4 = f(t + dt, y + dt*k3)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k3_it = k3.begin();
            auto temp_it = temp_state.begin();
            
            temp_it[i] = state_it[i] + dt * k3_it[i];
        }
        this->sys_(this->current_time_ + dt, temp_state, k4);
        
        // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto state_it = state.begin();
            auto k1_it = k1.begin();
            auto k2_it = k2.begin();
            auto k3_it = k3.begin();
            auto k4_it = k4.begin();
            
            state_it[i] = state_it[i] + dt * (k1_it[i] + 
                                            static_cast<value_type>(2) * k2_it[i] + 
                                            static_cast<value_type>(2) * k3_it[i] + 
                                            k4_it[i]) / static_cast<time_type>(6);
        }
        
        this->advance_time(dt);
    }
};