#pragma once

/**
 * @file standard_parallelism.hpp
 * @brief Examples showing how to use standard parallelism libraries with diffeq
 * 
 * This file demonstrates how to integrate standard C++ parallelism libraries
 * with the diffeq library instead of creating custom parallel abstractions.
 * 
 * Standard libraries demonstrated:
 * - std::execution policies (C++17/20)
 * - OpenMP (cross-platform)
 * - Intel TBB (threading building blocks)
 * - NVIDIA Thrust (GPU acceleration without custom kernels)
 */

#include <diffeq.hpp>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>

// Conditional includes for optional libraries
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TBB_AVAILABLE
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/task_scheduler_init.h>
#endif

#ifdef THRUST_AVAILABLE
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

namespace diffeq::examples {

/**
 * @brief Example: Parallel ODE integration using std::execution policies
 * 
 * This shows how to use standard C++17/20 execution policies with
 * existing diffeq integrators - no custom parallel classes needed!
 */
template<typename State, typename Time>
class StandardParallelODE {
public:
    
    /**
     * @brief Integrate multiple initial conditions in parallel using std::execution
     * 
     * @param system The ODE system function
     * @param initial_conditions Vector of initial states
     * @param dt Time step
     * @param steps Number of integration steps
     */
    template<typename System>
    static void integrate_multiple_conditions(
        System&& system,
        std::vector<State>& initial_conditions,
        Time dt,
        int steps
    ) {
        // Use standard C++17 parallel execution - no custom classes!
        std::for_each(std::execution::par_unseq, 
                     initial_conditions.begin(), 
                     initial_conditions.end(),
                     [&](State& state) {
                         auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                         for (int i = 0; i < steps; ++i) {
                             integrator.step(state, dt);
                         }
                     });
    }
    
    /**
     * @brief Parameter sweep using std::execution policies
     * 
     * Demonstrate flexibility beyond just initial conditions -
     * vary parameters, integrators, callbacks, etc.
     */
    template<typename System>
    static void parameter_sweep(
        System&& system_template,
        const State& initial_state,
        const std::vector<Time>& parameters,
        std::vector<State>& results,
        Time dt,
        int steps
    ) {
        results.resize(parameters.size());
        
        // Create indices for parallel processing
        std::vector<size_t> indices(parameters.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Parallel parameter sweep using standard library
        std::for_each(std::execution::par, 
                     indices.begin(), 
                     indices.end(),
                     [&](size_t i) {
                         State state = initial_state;
                         Time param = parameters[i];
                         
                         // Create system with this parameter
                         auto system = [&](const State& y, State& dydt, Time t) {
                             system_template(y, dydt, t, param);  // Pass parameter
                         };
                         
                         auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                         for (int step = 0; step < steps; ++step) {
                             integrator.step(state, dt);
                         }
                         results[i] = state;
                     });
    }
};

#ifdef _OPENMP
/**
 * @brief OpenMP examples for CPU parallelism
 * 
 * OpenMP is widely supported and doesn't require custom infrastructure.
 */
template<typename State, typename Time>
class OpenMPParallelODE {
public:
    
    template<typename System>
    static void integrate_openmp(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps
    ) {
        // Simple OpenMP parallel loop - no custom classes needed!
        #pragma omp parallel for
        for (size_t i = 0; i < states.size(); ++i) {
            auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
            for (int step = 0; step < steps; ++step) {
                integrator.step(states[i], dt);
            }
        }
    }
    
    /**
     * @brief Different integrators in parallel
     * Demonstrate running different algorithms simultaneously
     */
    template<typename System>
    static void multi_integrator_comparison(
        System&& system,
        const State& initial_state,
        Time dt,
        int steps,
        std::vector<State>& rk4_results,
        std::vector<State>& euler_results
    ) {
        const int num_runs = 100;
        rk4_results.resize(num_runs, initial_state);
        euler_results.resize(num_runs, initial_state);
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // RK4 integration
                #pragma omp parallel for
                for (int i = 0; i < num_runs; ++i) {
                    auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                    for (int step = 0; step < steps; ++step) {
                        integrator.step(rk4_results[i], dt);
                    }
                }
            }
            
            #pragma omp section
            {
                // Euler integration (for comparison)
                #pragma omp parallel for
                for (int i = 0; i < num_runs; ++i) {
                    auto integrator = diffeq::integrators::ode::EulerIntegrator<State, Time>(system);
                    for (int step = 0; step < steps; ++step) {
                        integrator.step(euler_results[i], dt);
                    }
                }
            }
        }
    }
};
#endif

#ifdef TBB_AVAILABLE
/**
 * @brief Intel TBB examples for advanced CPU parallelism
 * 
 * TBB provides sophisticated parallel algorithms and task scheduling.
 */
template<typename State, typename Time>
class TBBParallelODE {
public:
    template<typename System>
    static void integrate_tbb(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps
    ) {
        // Use TBB parallel_for_each - clean and efficient
        tbb::parallel_for_each(states.begin(), states.end(),
                              [&](State& state) {
                                  auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                                  for (int step = 0; step < steps; ++step) {
                                      integrator.step(state, dt);
                                  }
                              });
    }
    
    /**
     * @brief Blocked parallel execution for memory efficiency
     */
    template<typename System>
    static void integrate_blocked(
        System&& system,
        std::vector<State>& states,
        Time dt,
        int steps,
        size_t block_size = 1000
    ) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, states.size(), block_size),
                         [&](const tbb::blocked_range<size_t>& range) {
                             for (size_t i = range.begin(); i != range.end(); ++i) {
                                 auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                                 for (int step = 0; step < steps; ++step) {
                                     integrator.step(states[i], dt);
                                 }
                             }
                         });
    }
};
#endif

#ifdef THRUST_AVAILABLE
/**
 * @brief GPU acceleration using NVIDIA Thrust
 * 
 * Thrust provides GPU parallelism WITHOUT writing custom CUDA kernels!
 * The user specifically wanted to see how to run ODEs on GPU without
 * writing kernel functions - this is how.
 */
template<typename State, typename Time>
class ThrustGPUODE {
public:
    /**
     * @brief GPU ODE integration using Thrust - NO custom kernels needed!
     * 
     * This demonstrates exactly what the user asked for: GPU execution
     * without writing kernel functions.
     */
    template<typename System>
    static void integrate_gpu(
        System&& system,
        std::vector<State>& host_states,
        Time dt,
        int steps
    ) {
        // Copy data to GPU
        thrust::device_vector<State> device_states = host_states;
        
        // GPU parallel execution without custom kernels!
        thrust::for_each(thrust::device,
                        device_states.begin(),
                        device_states.end(),
                        [=] __device__ (State& state) {
                            // Note: This requires the integrator and system to be GPU-compatible
                            // For complex systems, you might need to adapt them for GPU
                            auto integrator = diffeq::integrators::ode::RK4Integrator<State, Time>(system);
                            for (int step = 0; step < steps; ++step) {
                                integrator.step(state, dt);
                            }
                        });
        
        // Copy results back to host
        thrust::copy(device_states.begin(), device_states.end(), host_states.begin());
    }
    
    /**
     * @brief Check GPU availability using standard CUDA functions
     * 
     * The user mentioned we shouldn't implement GPU detection ourselves -
     * use standard CUDA runtime functions instead.
     */
    static bool gpu_available() {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        return (error == cudaSuccess) && (device_count > 0);
    }
    
    /**
     * @brief Simple element-wise operations on GPU
     * 
     * For simpler mathematical operations, Thrust makes GPU usage trivial
     */
    static void transform_states_gpu(std::vector<State>& host_states,
                                   std::function<State(const State&)> transform) {
        thrust::device_vector<State> device_states = host_states;
        
        thrust::transform(thrust::device,
                         device_states.begin(),
                         device_states.end(),
                         device_states.begin(),
                         transform);
        
        thrust::copy(device_states.begin(), device_states.end(), host_states.begin());
    }
};
#endif

/**
 * @brief Utility functions for checking standard library availability
 * 
 * Instead of custom hardware detection, provide simple checks for
 * what standard libraries are available.
 */
namespace availability {
    
    inline bool std_execution_available() {
        #ifdef __cpp_lib_execution
        return true;
        #else
        return false;
        #endif
    }
    
    inline bool openmp_available() {
        #ifdef _OPENMP
        return true;
        #else
        return false;
        #endif
    }
    
    inline bool tbb_available() {
        #ifdef TBB_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }
    
    inline bool thrust_available() {
        #ifdef THRUST_AVAILABLE
        return true;
        #else
        return false;
        #endif
    }
}

} // namespace diffeq::examples