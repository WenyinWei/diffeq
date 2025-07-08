#pragma once
#include <functional>
#include <concepts>
#include <type_traits>
#include "concepts.hpp"

namespace diffeq::core {

// Abstract integrator base class
template<system_state S>
class AbstractIntegrator {
public:
    using state_type = S;
    using time_type = typename S::value_type;
    using value_type = typename S::value_type;
    using system_function = std::function<void(time_type, const state_type&, state_type&)>;

    explicit AbstractIntegrator(system_function sys) 
        : sys_(std::move(sys)), current_time_(time_type{0}) {}

    virtual ~AbstractIntegrator() = default;

    // Pure virtual function that derived classes must implement
    virtual void step(state_type& state, time_type dt) = 0;

    // Virtual function for multi-step integration (can be overridden)
    virtual void integrate(state_type& state, time_type dt, time_type end_time) {
        while (current_time_ < end_time) {
            time_type step_size = std::min(dt, end_time - current_time_);
            step(state, step_size);
        }
    }

    // Getters
    time_type current_time() const { return current_time_; }
    
    // Setters
    void set_time(time_type t) { current_time_ = t; }
    void set_system(system_function sys) { sys_ = std::move(sys); }

protected:
    system_function sys_;
    time_type current_time_;

    // Helper function for derived classes to update time
    void advance_time(time_type dt) { current_time_ += dt; }
};

} // namespace diffeq::core