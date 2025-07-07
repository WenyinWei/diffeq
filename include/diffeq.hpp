#pragma once

// Core concepts and base classes
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/adaptive_integrator.hpp>
#include <core/timeout_integrator.hpp>
#include <core/composable_integration.hpp>

// ODE integrator implementations (organized by method type)
#include <integrators/ode/euler.hpp>           // Simple Euler method
#include <integrators/ode/improved_euler.hpp>  // Heun's method
#include <integrators/ode/rk4.hpp>             // Classic 4th order Runge-Kutta
#include <integrators/ode/rk23.hpp>            // RK23 (adaptive, Bogacki-Shampine)
#include <integrators/ode/rk45.hpp>            // RK45 (adaptive, Dormand-Prince)
#include <integrators/ode/dop853.hpp>          // DOP853 (8th order, high accuracy)
#include <integrators/ode/bdf.hpp>             // BDF (multistep, stiff systems)
#include <integrators/ode/lsoda.hpp>           // LSODA (automatic stiff/non-stiff switching)

// SDE (Stochastic Differential Equation) integrators (organized by method type)
#include <sde/sde_base.hpp>                    // SDE base infrastructure
#include <integrators/sde/euler_maruyama.hpp>  // Basic SDE solver (strong order 0.5)
#include <integrators/sde/milstein.hpp>        // Milstein method with Lévy area (strong order 1.0)
#include <integrators/sde/sri1.hpp>            // Stochastic Runge-Kutta method (strong order 1.0)
#include <integrators/sde/implicit_euler_maruyama.hpp>  // Implicit method for stiff SDEs
#include <integrators/sde/sra.hpp>             // SRA base implementation
#include <integrators/sde/sra1.hpp>            // SRA1 variant for additive noise
#include <integrators/sde/sra2.hpp>            // SRA2 variant for additive noise
#include <integrators/sde/sosra.hpp>           // Stability-optimized SRA
#include <integrators/sde/sri.hpp>             // SRI base implementation
#include <integrators/sde/sriw1.hpp>           // SRIW1 variant for general SDEs
#include <integrators/sde/sosri.hpp>           // Stability-optimized SRI

// Modern async and signal processing components (standard C++ only)
#include <async/async_integrator.hpp>    // Async integration with std::future
#include <signal/signal_processor.hpp>   // Generic signal processing
#include <interfaces/integration_interface.hpp>  // Unified interface for all domains

// Standard parallelism library integration examples
// Note: Use standard libraries (std::execution, OpenMP, TBB, Thrust) instead of custom parallel classes
// See docs/STANDARD_PARALLELISM.md and examples/standard_parallelism_demo.cpp for integration examples

/**
 * @file diffeq.hpp
 * @brief Modern C++ ODE Integration Library with Real-time Signal Processing
 * 
 * This library provides a comprehensive C++20 implementation of ODE integrators
 * with advanced real-time capabilities for financial and robotics applications.
 * Features include real-time signal processing, inter-process communication,
 * and async execution using modern C++ standards.
 * 
 * Core Integrators:
 * ================
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
 * SDE (Stochastic Differential Equation) Integrators:
 * ===================================================
 * 
 * Basic Methods:
 * - EulerMaruyamaIntegrator: Basic SDE solver (strong order 0.5)
 * - MilsteinIntegrator: Higher-order method with Lévy area (strong order 1.0)
 * - SRI1Integrator: Stochastic Runge-Kutta method (strong order 1.0)
 * - ImplicitEulerMaruyamaIntegrator: Implicit method for stiff SDEs
 * 
 * Advanced High-Order Methods (Strong Order 1.5):
 * - SRAIntegrator: Stochastic Runge-Kutta for additive noise SDEs
 * - SRIIntegrator: Stochastic Runge-Kutta for general Itô SDEs  
 * - SOSRAIntegrator: Stability-optimized SRA, robust to stiffness
 * - SOSRIIntegrator: Stability-optimized SRI, robust to stiffness
 * 
 * Pre-configured Methods:
 * - SRA1, SRA2: Different tableau variants for additive noise
 * - SRIW1: Weak order 2.0 variant for general SDEs
 * - SOSRA, SOSRI: Stability-optimized for high tolerances
 * 
 * Real-time Capabilities:
 * ======================
 * 
 * - AsyncIntegrator: Lightweight async wrapper using std::future and std::thread
 * - SignalProcessor: Generic signal processing with type-safe handlers
 * - IntegrationInterface: Unified interface for signal-aware ODE integration
 * - Extensible design: Works for finance, robotics, science, and any domain
 * 
 * Key Features:
 * - Header-only design (no external dependencies)
 * - Standard C++20 facilities only with proper concepts
 * - Optional external library support (networking, JSON, etc.)
 * - Thread-safe async operations
 * - Unified interface for all application domains
 * 
 * Usage Examples:
 * ===============
 * 
 * Basic ODE Integration:
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
 *     std::vector<double> y = {1.0};  // Initial condition
 *     RK45Integrator<std::vector<double>> integrator(exponential_decay);
 *     integrator.integrate(y, 0.1, 1.0);  // Integrate from t=0 to t=1
 *     // Result: y[0] ≈ exp(-1) ≈ 0.368
 *     return 0;
 * }
 * ```
 * 
 * Real-time Signal-Aware Integration:
 * 
 * #include <diffeq.hpp>
 * #include <interfaces/integration_interface.hpp>
 * 
 * // Create signal-aware interface
 * auto interface = diffeq::interfaces::make_integration_interface<std::vector<double>>();
 * 
 * // Register signal influences
 * interface->register_signal_influence<double>("price_update",
 *     diffeq::interfaces::IntegrationInterface<std::vector<double>>::InfluenceMode::CONTINUOUS_SHIFT,
 *     [](const double& price, auto& state, auto t) {
 *         // Modify portfolio dynamics based on price signal
 *         double momentum = (price > 100.0) ? 0.01 : -0.01;
 *         for (auto& asset : state) asset *= (1.0 + momentum);
 *     });
 * 
 * // Register real-time output
 * interface->register_output_stream("monitor",
 *     [](const auto& state, auto t) {
 *         std::cout << "Portfolio value: " << std::accumulate(state.begin(), state.end(), 0.0) << std::endl;
 *     });
 * 
 * // Create signal-aware ODE
 * auto signal_ode = interface->make_signal_aware_ode(my_portfolio_ode);
 * auto integrator = diffeq::make_rk45<std::vector<double>>(signal_ode);
 * 
 * // Integration automatically handles signals and outputs
 * integrator.integrate(state, dt, t_final);
 * 
 * Real-time Robot Control:
 * 
 * #include <diffeq.hpp>
 * #include <interfaces/integration_interface.hpp>
 * 
 * constexpr size_t N_JOINTS = 6;
 * std::array<double, N_JOINTS * 3> robot_state{}; // position, velocity, acceleration
 * 
 * // Create robotics interface
 * auto interface = diffeq::interfaces::make_integration_interface<std::array<double, 18>>();
 * 
 * // Register control signal influence
 * interface->register_signal_influence<std::vector<double>>("control_targets",
 *     diffeq::interfaces::IntegrationInterface<std::array<double, 18>>::InfluenceMode::DISCRETE_EVENT,
 *     [](const auto& targets, auto& state, auto t) {
 *         // Update target positions for each joint
 *         for (size_t i = 0; i < targets.size() && i < N_JOINTS; ++i) {
 *             // Apply control logic
 *         }
 *     });
 * 
 * // Emergency stop capability
 * interface->register_signal_influence<bool>("emergency_stop",
 *     diffeq::interfaces::IntegrationInterface<std::array<double, 18>>::InfluenceMode::DISCRETE_EVENT,
 *     [](bool stop, auto& state, auto t) {
 *         if (stop) {
 *             // Set all velocities to zero
 *             for (size_t i = N_JOINTS; i < 2 * N_JOINTS; ++i) state[i] = 0.0;
 *         }
 *     });
 * 
 * // Create robotics interface and register signals for real-time control.
 * // Example shows emergency stop and joint monitoring capabilities.
 * 
 * Stochastic Differential Equations (SDEs):
 * ==========================================
 * 
 * ```cpp
 * #include <diffeq.hpp>
 * #include <sde/sde_base.hpp>
 * #include <sde/sde_solvers.hpp>
 * 
 * using namespace diffeq::sde;
 * 
 * // Define SDE system: dX = f(t,X)dt + g(t,X)dW
 * // Example: Geometric Brownian Motion dS = μS dt + σS dW
 * auto drift = [](double t, const std::vector<double>& x, std::vector<double>& fx) {
 *     double mu = 0.05;  // 5% drift
 *     fx[0] = mu * x[0];
 * };
 * 
 * auto diffusion = [](double t, const std::vector<double>& x, std::vector<double>& gx) {
 *     double sigma = 0.2;  // 20% volatility
 *     gx[0] = sigma * x[0];
 * };
 * 
 * // Create SDE problem and integrator
 * auto problem = factory::make_sde_problem<std::vector<double>, double>(
 *     drift, diffusion, NoiseType::DIAGONAL_NOISE);
 * auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 42);
 * 
 * EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
 * 
 * std::vector<double> state = {100.0};  // Initial stock price
 * integrator.integrate(state, 0.01, 1.0);  // Integrate to t=1
 * ```
 * 
 * Available SDE methods:
 * - EulerMaruyamaIntegrator: Basic method (strong order 0.5)
 * - MilsteinIntegrator: Higher accuracy with Lévy area (strong order 1.0)
 * - SRI1Integrator: Stochastic Runge-Kutta (strong order 1.0)
 * - ImplicitEulerMaruyamaIntegrator: For stiff SDEs
 * 
 * Guidelines for Integrator Selection:
 * =====================================
 * 
 * 1. **RK45Integrator**: Best general-purpose choice for smooth, non-stiff ODEs.
 *    Similar to scipy.integrate.solve_ivp with method RK45.
 * 
 * 2. **RK23Integrator**: Lower accuracy but faster for less demanding problems.
 *    Similar to scipy.integrate.solve_ivp with method RK23.
 * 
 * 3. **DOP853Integrator**: High accuracy for demanding smooth problems.
 *    Similar to scipy.integrate.solve_ivp with method DOP853.
 * 
 * 4. **BDFIntegrator**: Variable order method for stiff systems.
 *    Similar to scipy.integrate.solve_ivp with method BDF.
 * 
 * 5. **RadauIntegrator**: Implicit method for stiff systems.
 *    Similar to scipy.integrate.solve_ivp with method Radau.
 * 
 * 6. **LSODAIntegrator**: Automatic stiffness detection and method switching.
 *    Similar to scipy.integrate.odeint or solve_ivp with method LSODA.
 * 
 * 7. **RK4Integrator**: Simple fixed-step method for educational purposes or
 *    when step size control is handled externally.
 * 
 * Concepts:
 * =========
 * 
 * - system_state: Container types (std::vector, std::array, etc.) that can
 *   represent the state vector of the ODE system.
 * 
 * - can_be_time: Arithmetic types (double, float, int) that can represent time.
 * 
 * All integrators and interfaces are templated on these concepts for maximum
 * flexibility and type safety.
 * 
 * Architecture:
 * =============
 * 
 * The library uses a unified interface design where all real-time signal processing
 * capabilities are provided through a single IntegrationInterface class. This 
 * interface supports:
 * 
 * 1. **Discrete Events**: Instantaneous state modifications triggered by signals
 * 2. **Continuous Influences**: Ongoing trajectory modifications from signals  
 * 3. **Parameter Updates**: Dynamic changes to integration parameters
 * 4. **Real-time Output**: Streaming integration data at specified intervals
 * 
 * This design eliminates the need for domain-specific processors while remaining
 * flexible enough to handle any application domain.
 */

namespace diffeq {
    // Re-export commonly used types for convenience
    using std::vector;
    using std::array;
    
    // Re-export core functionality
    using core::TimeoutIntegrator;
    using core::IntegrationResult;
    using core::IntegrationTimeoutException;
    using core::make_timeout_integrator;
    using core::integrate_with_timeout;
    
    // Note: ParallelTimeoutIntegrator was removed in favor of composable architecture
    // Use make_builder(base).with_timeout().with_parallel().build() instead
    
    // Re-export composable integration facilities
    using core::composable::IntegratorDecorator;
    using core::composable::TimeoutDecorator;
    using core::composable::ParallelDecorator;
    using core::composable::AsyncDecorator;
    using core::composable::OutputDecorator;
    using core::composable::SignalDecorator;
    using core::composable::IntegratorBuilder;
    using core::composable::make_builder;
    using core::composable::with_timeout_only;
    using core::composable::with_parallel_only;
    using core::composable::with_async_only;
    using core::composable::TimeoutConfig;
    using core::composable::TimeoutResult;
    using core::composable::ParallelConfig;
    using core::composable::AsyncConfig;
    using core::composable::OutputConfig;
    using core::composable::OutputMode;
    using core::composable::SignalConfig;
    
    // Re-export integrator classes for convenience
    // Note: Integrators are already in diffeq namespace, no need to re-export
    
    // Re-export SDE integrators
    // Note: SDE integrators are already in diffeq namespace, no need to re-export
    
    // Common type aliases for system_state concept
    template<typename T>
    using VectorState = std::vector<T>;
    
    template<typename T, std::size_t N>
    using ArrayState = std::array<T, N>;
    
    // Default scalar types for can_be_time concept
    using DefaultScalar = double;
    using DefaultTime = double;
}


