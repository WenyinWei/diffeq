#pragma once

#include <diffeq.hpp>
#include <async/async_integrator.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <chrono>

namespace diffeq::examples::sde {

/**
 * @brief Financial Market Modeling with Advanced SDE Solvers
 * 
 * Demonstrates Black-Scholes model, Heston stochastic volatility,
 * and jump-diffusion processes using high-order SDE integrators.
 */
namespace finance {

/**
 * @brief Black-Scholes model: dS = μS dt + σS dW
 */
class BlackScholesModel {
public:
    double mu, sigma;  // drift and volatility
    
    BlackScholesModel(double mu = 0.05, double sigma = 0.2) 
        : mu(mu), sigma(sigma) {}
    
    void run_comparison() {
        std::cout << "\n=== Black-Scholes Model Comparison ===\n";
        std::cout << "μ = " << mu << ", σ = " << sigma << std::endl;
        
        auto drift_func = [this](double /*t*/, const std::vector<double>& S, std::vector<double>& dS) {
            dS[0] = mu * S[0];
        };
        
        auto diffusion_func = [this](double /*t*/, const std::vector<double>& S, std::vector<double>& gS) {
            gS[0] = sigma * S[0];
        };
        
        auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::sde::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>, double>(1, 12345);
        
        double S0 = 100.0;  // Initial stock price
        double T = 1.0;     // Time to maturity
        double dt = 0.01;   // Time step
        int steps = static_cast<int>(T / dt);
        
        // Analytical solution
        double analytical = S0 * std::exp(mu * T);
        std::cout << "Analytical expected value: " << analytical << std::endl;
        
        // Compare different methods
        std::vector<std::string> methods = {"Euler-Maruyama", "Milstein", "SRA1", "SOSRA", "SRIW1", "SOSRI"};
        std::vector<double> results;
        
        // Euler-Maruyama
        {
            diffeq::sde::EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // Milstein
        {
            diffeq::sde::MilsteinIntegrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SRA1
        {
            diffeq::sde::SRA1Integrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SOSRA
        {
            auto integrator = diffeq::sde::factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = {S0};
            integrator->set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SRIW1
        {
            auto integrator = diffeq::sde::factory::make_sriw1_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = {S0};
            integrator->set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SOSRI
        {
            auto integrator = diffeq::sde::factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = {S0};
            integrator->set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        for (size_t i = 0; i < methods.size(); ++i) {
            double error = std::abs(results[i] - analytical) / analytical * 100.0;
            std::cout << methods[i] << ": " << results[i] 
                      << " (error: " << error << "%)" << std::endl;
        }
    }
};

/**
 * @brief Heston stochastic volatility model
 * 
 * dS = μS dt + √V S dW1
 * dV = κ(θ - V) dt + σ √V dW2
 * 
 * Two-dimensional SDE with correlated noise
 */
class HestonModel {
public:
    double mu, kappa, theta, sigma, rho;
    
    HestonModel(double mu = 0.05, double kappa = 2.0, double theta = 0.04, 
                double sigma = 0.3, double rho = -0.7) 
        : mu(mu), kappa(kappa), theta(theta), sigma(sigma), rho(rho) {}
    
    void run_example() {
        std::cout << "\n=== Heston Stochastic Volatility Model ===\n";
        std::cout << "μ = " << mu << ", κ = " << kappa << ", θ = " << theta 
                  << ", σ = " << sigma << ", ρ = " << rho << std::endl;
        
        auto drift_func = [this](double /*t*/, const std::vector<double>& x, std::vector<double>& dx) {
            double S = x[0], V = x[1];
            dx[0] = mu * S;
            dx[1] = kappa * (theta - V);
        };
        
        auto diffusion_func = [this](double /*t*/, const std::vector<double>& x, std::vector<double>& gx) {
            double S = x[0], V = std::max(x[1], 0.0);  // Ensure V ≥ 0
            gx[0] = std::sqrt(V) * S;
            gx[1] = sigma * std::sqrt(V);
        };
        
        auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::sde::NoiseType::GENERAL_NOISE);
        
        auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>, double>(2, 54321);
        
        // Set correlated noise
        auto correlated_noise_func = [this](double /*t*/, const std::vector<double>& /*x*/, 
                                           std::vector<double>& noise_term, const std::vector<double>& dW) {
            // Apply correlation: dW2_corr = ρ*dW1 + √(1-ρ²)*dW2
            double dW1 = dW[0];
            double dW2_corr = rho * dW1 + std::sqrt(1 - rho*rho) * dW[1];
            
            noise_term[0] *= dW1;
            noise_term[1] *= dW2_corr;
        };
        
        problem->set_noise_function(correlated_noise_func);
        
        std::vector<double> x = {100.0, 0.04};  // Initial [S, V]
        double dt = 0.01;
        double T = 1.0;
        int steps = static_cast<int>(T / dt);
        
        // Use SOSRI for stability with this complex model
        auto integrator = diffeq::sde::factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
        
        integrator->set_time(0.0);
        
        std::cout << "Initial: S = " << x[0] << ", V = " << x[1] << std::endl;
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(x, dt);
            
            // Print intermediate results
            if (i % (steps/10) == 0) {
                std::cout << "t = " << (i+1) * dt << ": S = " << x[0] 
                          << ", V = " << x[1] << std::endl;
            }
        }
        
        std::cout << "Final: S = " << x[0] << ", V = " << x[1] << std::endl;
    }
};

} // namespace finance

/**
 * @brief Engineering Applications: Control Systems with Noise
 */
namespace engineering {

/**
 * @brief Noisy oscillator with control
 * 
 * ẍ + 2ζωₙẋ + ωₙ²x = u + σ ξ(t)
 * 
 * Where ξ(t) is white noise, u is control input
 */
class NoisyOscillator {
public:
    double omega_n, zeta, sigma;
    std::function<double(double, double, double)> control_law;
    
    NoisyOscillator(double omega_n = 1.0, double zeta = 0.1, double sigma = 0.1)
        : omega_n(omega_n), zeta(zeta), sigma(sigma) {
        
        // Default PD control: u = -Kp*x - Kd*ẋ
        control_law = [](double t, double x, double xdot) {
            double Kp = 2.0, Kd = 1.0;
            return -Kp * x - Kd * xdot;
        };
    }
    
    void run_control_example() {
        std::cout << "\n=== Noisy Oscillator Control ===\n";
        std::cout << "ωₙ = " << omega_n << ", ζ = " << zeta << ", σ = " << sigma << std::endl;
        
        auto drift_func = [this](double t, const std::vector<double>& state, std::vector<double>& dstate) {
            double x = state[0], xdot = state[1];
            double u = control_law(t, x, xdot);
            
            dstate[0] = xdot;
            dstate[1] = -2*zeta*omega_n*xdot - omega_n*omega_n*x + u;
        };
        
        auto diffusion_func = [this](double t, const std::vector<double>& state, std::vector<double>& gstate) {
            gstate[0] = 0.0;      // No noise on position directly
            gstate[1] = sigma;    // Additive noise on acceleration
        };
        
        auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::sde::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>, double>(2, 98765);
        
        // Use SOSRA since we have additive noise
        auto integrator = diffeq::sde::factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
        
        std::vector<double> state = {1.0, 0.0};  // Initial [position, velocity]
        double dt = 0.01;
        double T = 5.0;
        int steps = static_cast<int>(T / dt);
        
        integrator->set_time(0.0);
        
        std::cout << "Initial: x = " << state[0] << ", ẋ = " << state[1] << std::endl;
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(state, dt);
            
            if (i % (steps/10) == 0) {
                double u = control_law(i * dt, state[0], state[1]);
                std::cout << "t = " << (i+1) * dt << ": x = " << state[0] 
                          << ", ẋ = " << state[1] << ", u = " << u << std::endl;
            }
        }
        
        std::cout << "Final: x = " << state[0] << ", ẋ = " << state[1] << std::endl;
    }
};

} // namespace engineering

/**
 * @brief Scientific Computing: Population Dynamics with Environmental Noise
 */
namespace science {

/**
 * @brief Stochastic Lotka-Volterra predator-prey model
 * 
 * dx = (α - βy)x dt + σ₁x dW₁
 * dy = (δx - γ)y dt + σ₂y dW₂
 */
class StochasticLotkaVolterra {
public:
    double alpha, beta, gamma, delta, sigma1, sigma2;
    
    StochasticLotkaVolterra(double alpha = 1.0, double beta = 1.0, double gamma = 1.0, 
                           double delta = 1.0, double sigma1 = 0.1, double sigma2 = 0.1)
        : alpha(alpha), beta(beta), gamma(gamma), delta(delta), sigma1(sigma1), sigma2(sigma2) {}
    
    void run_ecosystem_simulation() {
        std::cout << "\n=== Stochastic Lotka-Volterra Ecosystem ===\n";
        std::cout << "α = " << alpha << ", β = " << beta << ", γ = " << gamma 
                  << ", δ = " << delta << std::endl;
        std::cout << "σ₁ = " << sigma1 << ", σ₂ = " << sigma2 << std::endl;
        
        auto drift_func = [this](double t, const std::vector<double>& pop, std::vector<double>& dpop) {
            double x = pop[0], y = pop[1];  // prey, predator populations
            dpop[0] = (alpha - beta * y) * x;
            dpop[1] = (delta * x - gamma) * y;
        };
        
        auto diffusion_func = [this](double t, const std::vector<double>& pop, std::vector<double>& gpop) {
            double x = pop[0], y = pop[1];
            gpop[0] = sigma1 * x;
            gpop[1] = sigma2 * y;
        };
        
        auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::sde::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>, double>(2, 13579);
        
        // Use SRIW1 for good weak order performance in this ecological model
        auto integrator = diffeq::sde::factory::make_sriw1_integrator<std::vector<double>, double>(problem, wiener);
        
        std::vector<double> population = {10.0, 5.0};  // Initial [prey, predator]
        double dt = 0.01;
        double T = 10.0;
        int steps = static_cast<int>(T / dt);
        
        integrator->set_time(0.0);
        
        std::cout << "Initial populations: Prey = " << population[0] << ", Predator = " << population[1] << std::endl;
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(population, dt);
            
            // Ensure populations stay positive (simple reflection)
            population[0] = std::max(population[0], 0.1);
            population[1] = std::max(population[1], 0.1);
            
            if (i % (steps/20) == 0) {
                std::cout << "t = " << (i+1) * dt << ": Prey = " << population[0] 
                          << ", Predator = " << population[1] << std::endl;
            }
        }
        
        std::cout << "Final populations: Prey = " << population[0] << ", Predator = " << population[1] << std::endl;
    }
};

} // namespace science

/**
 * @brief Unified Interface Usage with SDEs
 */
namespace unified_interface {

void demonstrate_async_sde_integration() {
    std::cout << "\n=== Async SDE Integration with Unified Interface ===\n";
    
    // Create a simple SDE problem
    auto drift_func = [](double /*t*/, const std::vector<double>& x, std::vector<double>& dx) {
        dx[0] = -0.5 * x[0];  // Mean-reverting process
    };
    
    auto diffusion_func = [](double /*t*/, const std::vector<double>& /*x*/, std::vector<double>& gx) {
        gx[0] = 0.2;  // Additive noise
    };
    
    auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, diffeq::sde::NoiseType::DIAGONAL_NOISE);
    
    auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>, double>(1, 24680);
    
    // Create SOSRA integrator for robust performance
    auto integrator = diffeq::sde::factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
    
    std::vector<double> initial_state = {1.0};
    double dt = 0.01;
    double end_time = 1.0;
    
    std::cout << "Starting SDE integration..." << std::endl;
    
    // Simulate async-style integration (the integrator can be wrapped in std::async)
    auto start_time = std::chrono::steady_clock::now();
    
    std::vector<double> state = initial_state;
    integrator->set_time(0.0);
    
    while (integrator->current_time() < end_time) {
        double step_size = std::min(dt, end_time - integrator->current_time());
        integrator->step(state, step_size);
    }
    
    auto end_time_chrono = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_chrono - start_time);
    
    std::cout << "SDE integration complete. Final value: " << state[0] << std::endl;
    std::cout << "Integration time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Note: This can be easily wrapped with std::async for true async execution" << std::endl;
}

void demonstrate_signal_aware_sde() {
    std::cout << "\n=== Signal-Aware SDE Integration ===\n";
    
    // This could be used for real-time financial data processing,
    // robotics control with sensor updates, etc.
    
    std::cout << "Signal-aware SDE integration ready for real-time applications:\n";
    std::cout << "• Financial: Real-time option pricing with market data feeds\n";
    std::cout << "• Robotics: State estimation with sensor noise and control updates\n";
    std::cout << "• Science: Environmental monitoring with stochastic processes\n";
    std::cout << "• All methods support the unified IntegrationInterface\n";
}

} // namespace unified_interface

/**
 * @brief Performance and accuracy comparison
 */
void run_comprehensive_sde_examples() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "    COMPREHENSIVE SDE SOLVER DEMONSTRATION" << std::endl;
    std::cout << "    Enhanced C++ Implementation with DifferentialEquations.jl algorithms" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Financial applications
    finance::BlackScholesModel bs_model(0.05, 0.2);
    bs_model.run_comparison();
    
    finance::HestonModel heston_model;
    heston_model.run_example();
    
    // Engineering applications  
    engineering::NoisyOscillator oscillator;
    oscillator.run_control_example();
    
    // Scientific applications
    science::StochasticLotkaVolterra ecosystem;
    ecosystem.run_ecosystem_simulation();
    
    // Unified interface capabilities
    unified_interface::demonstrate_async_sde_integration();
    unified_interface::demonstrate_signal_aware_sde();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ All SDE examples completed successfully!" << std::endl;
    std::cout << "\n🔬 Implemented High-Order SDE Methods:" << std::endl;
    std::cout << "• SRA family: Strong order 1.5 for additive noise (SRA1, SRA2, SOSRA)" << std::endl;
    std::cout << "• SRI family: Strong order 1.5 for general Itô SDEs (SRIW1, SOSRI)" << std::endl;
    std::cout << "• Stability-optimized variants (SOSRA, SOSRI) for robust performance" << std::endl;
    std::cout << "• Proper tableau-based implementation following DifferentialEquations.jl" << std::endl;
    std::cout << "\n🚀 Modern C++ Features:" << std::endl;
    std::cout << "• Concept-based design with proper type safety" << std::endl;
    std::cout << "• Async integration support for real-time applications" << std::endl;
    std::cout << "• Unified interface for cross-domain usage" << std::endl;
    std::cout << "• Header-only, zero external dependencies" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

} // namespace diffeq::examples::sde
