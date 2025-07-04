#pragma once

// Core concepts and base classes
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/adaptive_integrator.hpp>

// All integrator implementations
#include <solvers/euler_solvers.hpp>     // Euler, ImprovedEuler, etc.
#include <solvers/rk4_solver.hpp>        // RK4 (fixed step)
#include <solvers/rk23_solver.hpp>       // RK23 (adaptive, Bogacki-Shampine)
#include <solvers/rk45_solver.hpp>       // RK45 (adaptive, Dormand-Prince)
#include <solvers/dop853_solver.hpp>     // DOP853 (8th order, high accuracy)
#include <solvers/radau_solver.hpp>      // Radau IIA (implicit, stiff systems)
#include <solvers/bdf_solver.hpp>        // BDF (multistep, stiff systems)
#include <solvers/lsoda_solver.hpp>      // LSODA (automatic method switching)

/**
 * @file diffeq.hpp
 * @brief Comprehensive ODE integrator library header
 * 
 * This library provides a modern C++20 implementation of popular ODE integrators
 * similar to those found in scipy.integrate. All integrators follow a consistent
 * API and support both fixed-size containers (std::array) and dynamic containers
 * (std::vector).
 * 
 * Available Integrators:
 * =====================
 * 
 * Fixed Step Methods:
 * - EulerIntegrator: Simple 1st order explicit method
 * - ImprovedEulerIntegrator: 2nd order Heun's method
 * - RK4Integrator: Classic 4th order Runge-Kutta
 * 
 * Adaptive Step Methods (Non-stiff):
 * - RK23Integrator: 3rd order Bogacki-Shampine with error control
 * - RK45Integrator: 5th order Dormand-Prince (scipy's default)
 * - DOP853Integrator: 8th order high-accuracy method
 * 
 * Stiff System Methods:
 * - RadauIntegrator: 5th order implicit Runge-Kutta
 * - BDFIntegrator: Variable order (1-6) backward differentiation
 * 
 * Automatic Methods:
 * - LSODAIntegrator: Automatic switching between Adams and BDF
 * 
 * Usage Example:
 * ==============
 * 
 * ```cpp
 * #include <diffeq.hpp>
 * #include <vector>
 * 
 * // Define ODE system: dy/dt = -y
 * void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
 *     dydt[0] = -y[0];
 * }
 * 
 * int main() {
 *     // Choose integrator based on problem characteristics:
 *     
 *     // For smooth, non-stiff problems - use RK45 (scipy default)
 *     RK45Integrator<std::vector<double>> integrator(exponential_decay);
 *     
 *     // For high accuracy - use DOP853
 *     // DOP853Integrator<std::vector<double>> integrator(exponential_decay, 1e-12, 1e-15);
 *     
 *     // For stiff problems - use BDF or Radau
 *     // BDFIntegrator<std::vector<double>> integrator(exponential_decay);
 *     
 *     // For unknown stiffness - use LSODA (automatic)
 *     // LSODAIntegrator<std::vector<double>> integrator(exponential_decay);
 *     
 *     std::vector<double> y = {1.0};  // Initial condition
 *     integrator.set_time(0.0);       // Set initial time
 *     integrator.integrate(y, 0.1, 1.0);  // Integrate from t=0 to t=1 with suggested dt=0.1
 *     
 *     // Result: y[0] ≈ exp(-1) ≈ 0.368
 *     return 0;
 * }
 * ```
 * 
 * Guidelines for Integrator Selection:
 * ===================================
 * 
 * 1. **RK45Integrator**: Best general-purpose choice for smooth, non-stiff ODEs.
 *    Similar to scipy.integrate.solve_ivp with method='RK45'.
 * 
 * 2. **RK23Integrator**: Lower accuracy but faster for less demanding problems.
 *    Similar to scipy.integrate.solve_ivp with method='RK23'.
 * 
 * 3. **DOP853Integrator**: High accuracy for demanding smooth problems.
 *    Similar to scipy.integrate.solve_ivp with method='DOP853'.
 * 
 * 4. **BDFIntegrator**: Variable order method for stiff systems.
 *    Similar to scipy.integrate.solve_ivp with method='BDF'.
 * 
 * 5. **RadauIntegrator**: Implicit method for stiff systems.
 *    Similar to scipy.integrate.solve_ivp with method='Radau'.
 * 
 * 6. **LSODAIntegrator**: Automatic stiffness detection and method switching.
 *    Similar to scipy.integrate.odeint or solve_ivp with method='LSODA'.
 * 
 * 7. **RK4Integrator**: Simple fixed-step method for educational purposes or
 *    when step size control is handled externally.
 * 
 * Concepts:
 * =========
 * 
 * - `system_state`: Container types (std::vector, std::array, etc.) that can
 *   represent the state vector of the ODE system.
 * 
 * - `can_be_time`: Arithmetic types (double, float, int) that can represent time.
 * 
 * All integrators are templated on these concepts for maximum flexibility.
 */

namespace diffeq {
    // Re-export commonly used types for convenience
    using std::vector;
    using std::array;
    
    // Common type aliases
    template<typename T>
    using VectorState = std::vector<T>;
    
    template<typename T, std::size_t N>
    using ArrayState = std::array<T, N>;
    
    // Default scalar types
    using DefaultScalar = double;
    using DefaultTime = double;
}

// Optional: Convenience factory functions
namespace diffeq {
    
    /**
     * Create an RK45 integrator (recommended default)
     */
    template<system_state S>
    auto make_rk45(typename AbstractIntegrator<S>::system_function sys,
                   typename S::value_type rtol = 1e-6,
                   typename S::value_type atol = 1e-9) {
        return RK45Integrator<S>(std::move(sys), rtol, atol);
    }
    
    /**
     * Create a high-accuracy DOP853 integrator
     */
    template<system_state S>
    auto make_dop853(typename AbstractIntegrator<S>::system_function sys,
                     typename S::value_type rtol = 1e-10,
                     typename S::value_type atol = 1e-15) {
        return DOP853Integrator<S>(std::move(sys), rtol, atol);
    }
    
    /**
     * Create a BDF integrator for stiff systems
     */
    template<system_state S>
    auto make_bdf(typename AbstractIntegrator<S>::system_function sys,
                  typename S::value_type rtol = 1e-6,
                  typename S::value_type atol = 1e-9) {
        return BDFIntegrator<S>(std::move(sys), rtol, atol);
    }
    
    /**
     * Create an LSODA integrator with automatic method switching
     */
    template<system_state S>
    auto make_lsoda(typename AbstractIntegrator<S>::system_function sys,
                    typename S::value_type rtol = 1e-6,
                    typename S::value_type atol = 1e-9) {
        return LSODAIntegrator<S>(std::move(sys), rtol, atol);
    }
}
