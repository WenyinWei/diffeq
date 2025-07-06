#pragma once

#include <core/concepts.hpp>
#include <functional>
#include <memory>
#include <random>
#include <chrono>

namespace diffeq::sde {

/**
 * @brief Noise process types for SDEs
 */
enum class NoiseType {
    SCALAR_NOISE,     // Single noise source
    DIAGONAL_NOISE,   // Independent noise for each component
    GENERAL_NOISE     // Correlated noise (full noise matrix)
};

/**
 * @brief SDE problem definition
 * 
 * Represents SDE of the form:
 * dX = f(t, X) dt + g(t, X) dW
 * 
 * Where:
 * - f is the drift function
 * - g is the diffusion function  
 * - dW is the Wiener process (Brownian motion)
 */
template<system_state StateType>
class SDEProblem {
public:
    using state_type = StateType;
    using time_type = typename StateType::value_type;
    using value_type = typename StateType::value_type;
    
    // Function signatures
    using drift_function = std::function<void(time_type, const state_type&, state_type&)>;
    using diffusion_function = std::function<void(time_type, const state_type&, state_type&)>;
    using noise_function = std::function<void(time_type, const state_type&, StateType&, const StateType&)>;
    
    SDEProblem(drift_function drift, diffusion_function diffusion, 
               NoiseType noise_type = NoiseType::DIAGONAL_NOISE)
        : drift_(std::move(drift))
        , diffusion_(std::move(diffusion))
        , noise_type_(noise_type) {}
    
    void drift(time_type t, const state_type& x, state_type& fx) const {
        drift_(t, x, fx);
    }
    
    void diffusion(time_type t, const state_type& x, state_type& gx) const {
        diffusion_(t, x, gx);
    }
    
    NoiseType get_noise_type() const { return noise_type_; }
    
    void set_noise_function(noise_function noise) {
        noise_ = std::move(noise);
    }
    
    bool has_custom_noise() const { return noise_ != nullptr; }
    
    void apply_noise(time_type t, const state_type& x, state_type& noise_term, const state_type& dW) const {
        if (noise_) {
            noise_(t, x, noise_term, dW);
        } else {
            // Default noise application based on noise type
            apply_default_noise(noise_term, dW);
        }
    }

private:
    drift_function drift_;
    diffusion_function diffusion_;
    noise_function noise_;
    NoiseType noise_type_;
    
    void apply_default_noise(state_type& noise_term, const state_type& dW) const {
        switch (noise_type_) {
            case NoiseType::SCALAR_NOISE:
                // All components use the same noise
                for (size_t i = 0; i < noise_term.size(); ++i) {
                    noise_term[i] *= dW[0];
                }
                break;
                
            case NoiseType::DIAGONAL_NOISE:
                // Each component has independent noise
                for (size_t i = 0; i < noise_term.size() && i < dW.size(); ++i) {
                    noise_term[i] *= dW[i];
                }
                break;
                
            case NoiseType::GENERAL_NOISE:
                // Custom noise - should be handled by noise function
                // Default to diagonal for safety
                for (size_t i = 0; i < noise_term.size() && i < dW.size(); ++i) {
                    noise_term[i] *= dW[i];
                }
                break;
        }
    }
};

/**
 * @brief Wiener process (Brownian motion) generator
 */
template<system_state StateType>
class WienerProcess {
public:
    using state_type = StateType;
    using time_type = typename StateType::value_type;
    using value_type = typename StateType::value_type;
    
    explicit WienerProcess(size_t dimension, uint32_t seed = 0)
        : dimension_(dimension)
        , generator_(seed == 0 ? std::chrono::steady_clock::now().time_since_epoch().count() : seed)
        , normal_dist_(0.0, 1.0) {}
    
    void generate_increment(state_type& dW, time_type dt) {
        value_type sqrt_dt = std::sqrt(static_cast<value_type>(dt));
        
        for (size_t i = 0; i < dimension_ && i < dW.size(); ++i) {
            auto dW_it = dW.begin();
            dW_it[i] = static_cast<value_type>(normal_dist_(generator_)) * sqrt_dt;
        }
    }
    
    void set_seed(uint32_t seed) {
        generator_.seed(seed);
    }
    
    size_t dimension() const { return dimension_; }

private:
    size_t dimension_;
    std::mt19937 generator_;
    std::normal_distribution<double> normal_dist_;
};

/**
 * @brief Abstract base class for SDE integrators
 */
template<system_state StateType>
class AbstractSDEIntegrator {
public:
    using state_type = StateType;
    using time_type = typename StateType::value_type;
    using value_type = typename StateType::value_type;
    using sde_problem_type = SDEProblem<StateType>;
    using wiener_process_type = WienerProcess<StateType>;
    
    explicit AbstractSDEIntegrator(std::shared_ptr<sde_problem_type> problem,
                                  std::shared_ptr<wiener_process_type> wiener = nullptr)
        : problem_(problem)
        , wiener_(wiener ? wiener : std::make_shared<wiener_process_type>(get_default_dimension(), 0))
        , current_time_(0) {}
    
    virtual ~AbstractSDEIntegrator() = default;
    
    // Pure virtual methods to be implemented by derived classes
    virtual void step(state_type& state, time_type dt) = 0;
    virtual std::string name() const = 0;
    
    // Integration interface
    void integrate(state_type& state, time_type dt, time_type end_time) {
        while (current_time_ < end_time) {
            time_type step_size = std::min(dt, end_time - current_time_);
            step(state, step_size);
        }
    }
    
    // Accessors
    time_type current_time() const { return current_time_; }
    void set_time(time_type t) { current_time_ = t; }
    
    std::shared_ptr<sde_problem_type> get_problem() const { return problem_; }
    std::shared_ptr<wiener_process_type> get_wiener_process() const { return wiener_; }
    
    void set_wiener_process(std::shared_ptr<wiener_process_type> wiener) {
        wiener_ = wiener;
    }

protected:
    void advance_time(time_type dt) { current_time_ += dt; }
    
    virtual size_t get_default_dimension() {
        // Default to assuming state dimension equals noise dimension
        return 10; // Will be overridden by actual state size in practice
    }

    std::shared_ptr<sde_problem_type> problem_;
    std::shared_ptr<wiener_process_type> wiener_;
    time_type current_time_;
};

/**
 * @brief Factory functions for creating SDE problems and Wiener processes
 */
namespace factory {

template<system_state StateType>
auto make_sde_problem(
    typename SDEProblem<StateType>::drift_function drift,
    typename SDEProblem<StateType>::diffusion_function diffusion,
    NoiseType noise_type = NoiseType::DIAGONAL_NOISE) {
    return std::make_shared<SDEProblem<StateType>>(std::move(drift), std::move(diffusion), noise_type);
}

template<system_state StateType>
auto make_wiener_process(size_t dimension, uint32_t seed = 0) {
    return std::make_shared<WienerProcess<StateType>>(dimension, seed);
}

} // namespace factory

} // namespace diffeq::sde
