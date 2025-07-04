#pragma once
#include <core/concepts.hpp>

// Helper trait to create state objects - shared among all integrators
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
