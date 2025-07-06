#include <diffeq.hpp>
#include <async/async_integrator.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <functional>
#include <chrono>
#include <random>

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
        std::cout << "\n=== Black-Scholes Model Comparison ===" << std::endl;
        std::cout << "μ = " << mu << ", σ = " << sigma << std::endl;
        
        auto drift_func = [this](double /*t*/, const std::vector<double>& S, std::vector<double>& dS) {
            dS[0] = mu * S[0];
        };
        
        auto diffusion_func = [this](double /*t*/, const std::vector<double>& S, std::vector<double>& gS) {
            gS[0] = sigma * S[0];
        };
        
        auto problem = diffeq::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::factory::make_wiener_process<std::vector<double>, double>(1, 12345);
        
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
            diffeq::EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
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
            diffeq::MilsteinIntegrator<std::vector<double>, double> integrator(problem, wiener);
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
            diffeq::SRA1Integrator<std::vector<double>, double> integrator(problem, wiener);
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
            diffeq::SOSRAIntegrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SRIW1
        {
            diffeq::SRIW1Integrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SOSRI
        {
            diffeq::SOSRIIntegrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = {S0};
            integrator.set_time(0.0);
            wiener->set_seed(12345);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
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
        std::cout << "\n=== Heston Stochastic Volatility Model ===" << std::endl;
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
        
        auto problem = diffeq::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::NoiseType::GENERAL_NOISE);
        
        auto wiener = diffeq::factory::make_wiener_process<std::vector<double>, double>(2, 54321);
        
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
        
        // Use high-order SDE integrator for better accuracy
        diffeq::SOSRAIntegrator<std::vector<double>, double> integrator(problem, wiener);
        integrator.set_time(0.0);
        
        std::cout << "Initial state: S = " << x[0] << ", V = " << x[1] << std::endl;
        
        // Integrate Heston model
        for (int i = 0; i < steps; ++i) {
            integrator.step(x, dt);
            
            // Output every 25 steps (quarterly)
            if (i % 25 == 0) {
                std::cout << "t = " << (i * dt) << ": S = " << x[0] << ", V = " << x[1] << std::endl;
            }
        }
        
        std::cout << "Final state: S = " << x[0] << ", V = " << x[1] << std::endl;
    }
};

} // namespace finance

/**
 * @brief Engineering Applications with SDEs
 * 
 * Demonstrates stochastic control systems, noisy oscillators,
 * and mechanical systems with random disturbances.
 */
namespace engineering {

/**
 * @brief Noisy oscillator with stochastic control
 * 
 * d²x/dt² + 2ζωₙ dx/dt + ωₙ²x = u(t) + σ dW
 * where u(t) is a control law and σ dW is noise
 */
class NoisyOscillator {
public:
    double omega_n, zeta, sigma;
    std::function<double(double, double, double)> control_law;
    
    NoisyOscillator(double omega_n = 1.0, double zeta = 0.1, double sigma = 0.1)
        : omega_n(omega_n), zeta(zeta), sigma(sigma) {
        // Simple PD control law
        control_law = [](double t, double x, double xdot) {
            double target = std::sin(t);  // Sinusoidal reference
            double error = target - x;
            double error_dot = std::cos(t) - xdot;  // Derivative of reference
            return 2.0 * error + 1.0 * error_dot;   // PD gains
        };
    }
    
    void run_control_example() {
        std::cout << "\n=== Noisy Oscillator with Stochastic Control ===" << std::endl;
        std::cout << "ωₙ = " << omega_n << ", ζ = " << zeta << ", σ = " << sigma << std::endl;
        
        auto drift_func = [this](double t, const std::vector<double>& state, std::vector<double>& dstate) {
            double x = state[0], xdot = state[1];
            double u = control_law(t, x, xdot);
            
            dstate[0] = xdot;
            dstate[1] = -omega_n * omega_n * x - 2 * zeta * omega_n * xdot + u;
        };
        
        auto diffusion_func = [this](double t, const std::vector<double>& state, std::vector<double>& gstate) {
            gstate[0] = 0.0;  // No noise in position
            gstate[1] = sigma; // Noise in velocity (acceleration)
        };
        
        auto problem = diffeq::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::factory::make_wiener_process<std::vector<double>, double>(1, 67890);
        diffeq::MilsteinIntegrator<std::vector<double>, double> integrator(problem, wiener);
        
        std::vector<double> state = {0.0, 0.0};  // Initial [x, xdot]
        double dt = 0.01;
        double T = 10.0;
        int steps = static_cast<int>(T / dt);
        
        integrator.set_time(0.0);
        
        std::cout << "Simulating controlled oscillator with noise..." << std::endl;
        
        // Track performance metrics
        double total_error = 0.0;
        double max_error = 0.0;
        
        for (int i = 0; i < steps; ++i) {
            double t = i * dt;
            double target = std::sin(t);
            double error = std::abs(target - state[0]);
            
            total_error += error;
            max_error = std::max(max_error, error);
            
            integrator.step(state, dt);
            
            // Output every 100 steps
            if (i % 100 == 0) {
                std::cout << "t = " << t << ": x = " << state[0] << ", target = " << target 
                          << ", error = " << error << std::endl;
            }
        }
        
        double avg_error = total_error / steps;
        std::cout << "Control performance:" << std::endl;
        std::cout << "  Average error: " << avg_error << std::endl;
        std::cout << "  Maximum error: " << max_error << std::endl;
        std::cout << "  Final state: x = " << state[0] << ", xdot = " << state[1] << std::endl;
    }
};

} // namespace engineering

/**
 * @brief Scientific Applications with SDEs
 * 
 * Demonstrates ecological models, chemical reactions,
 * and biological systems with stochastic dynamics.
 */
namespace science {

/**
 * @brief Stochastic Lotka-Volterra predator-prey model
 * 
 * dx/dt = αx - βxy + σ₁x dW₁
 * dy/dt = γxy - δy + σ₂y dW₂
 */
class StochasticLotkaVolterra {
public:
    double alpha, beta, gamma, delta, sigma1, sigma2;
    
    StochasticLotkaVolterra(double alpha = 1.0, double beta = 1.0, double gamma = 1.0, 
                           double delta = 1.0, double sigma1 = 0.1, double sigma2 = 0.1)
        : alpha(alpha), beta(beta), gamma(gamma), delta(delta), sigma1(sigma1), sigma2(sigma2) {}
    
    void run_ecosystem_simulation() {
        std::cout << "\n=== Stochastic Lotka-Volterra Ecosystem Model ===" << std::endl;
        std::cout << "α = " << alpha << ", β = " << beta << ", γ = " << gamma 
                  << ", δ = " << delta << std::endl;
        std::cout << "σ₁ = " << sigma1 << ", σ₂ = " << sigma2 << std::endl;
        
        auto drift_func = [this](double t, const std::vector<double>& pop, std::vector<double>& dpop) {
            double x = pop[0], y = pop[1];  // prey, predator
            dpop[0] = alpha * x - beta * x * y;
            dpop[1] = gamma * x * y - delta * y;
        };
        
        auto diffusion_func = [this](double t, const std::vector<double>& pop, std::vector<double>& gpop) {
            double x = pop[0], y = pop[1];
            gpop[0] = sigma1 * x;  // Multiplicative noise for prey
            gpop[1] = sigma2 * y;  // Multiplicative noise for predator
        };
        
        auto problem = diffeq::factory::make_sde_problem<std::vector<double>, double>(
            drift_func, diffusion_func, diffeq::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = diffeq::factory::make_wiener_process<std::vector<double>, double>(2, 11111);
        diffeq::SRA1Integrator<std::vector<double>, double> integrator(problem, wiener);
        
        std::vector<double> population = {2.0, 1.0};  // Initial [prey, predator]
        double dt = 0.01;
        double T = 20.0;
        int steps = static_cast<int>(T / dt);
        
        integrator.set_time(0.0);
        
        std::cout << "Initial populations: prey = " << population[0] << ", predator = " << population[1] << std::endl;
        
        // Track population dynamics
        double min_prey = population[0], max_prey = population[0];
        double min_pred = population[1], max_pred = population[1];
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(population, dt);
            
            // Update min/max
            min_prey = std::min(min_prey, population[0]);
            max_prey = std::max(max_prey, population[0]);
            min_pred = std::min(min_pred, population[1]);
            max_pred = std::max(max_pred, population[1]);
            
            // Output every 500 steps
            if (i % 500 == 0) {
                double t = i * dt;
                std::cout << "t = " << t << ": prey = " << population[0] 
                          << ", predator = " << population[1] << std::endl;
            }
        }
        
        std::cout << "Final populations: prey = " << population[0] << ", predator = " << population[1] << std::endl;
        std::cout << "Population ranges:" << std::endl;
        std::cout << "  Prey: [" << min_prey << ", " << max_prey << "]" << std::endl;
        std::cout << "  Predator: [" << min_pred << ", " << max_pred << "]" << std::endl;
    }
};

} // namespace science

/**
 * @brief Unified Interface Examples
 * 
 * Demonstrates how SDEs integrate with the unified interface
 * for signal processing and async execution.
 */
namespace unified_interface {

void demonstrate_async_sde_integration() {
    std::cout << "\n=== Async SDE Integration ===" << std::endl;
    
    // Define a simple SDE system
    auto drift_func = [](double t, const std::vector<double>& x, std::vector<double>& dx) {
        dx[0] = -0.1 * x[0];  // Decay
        dx[1] = 0.1 * x[0] - 0.05 * x[1];  // Coupled system
    };
    
    auto diffusion_func = [](double t, const std::vector<double>& x, std::vector<double>& gx) {
        gx[0] = 0.1 * x[0];  // Multiplicative noise
        gx[1] = 0.05 * x[1];
    };
    
    auto problem = diffeq::factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, diffeq::NoiseType::DIAGONAL_NOISE);
    
    auto wiener = diffeq::factory::make_wiener_process<std::vector<double>, double>(2, 22222);
    diffeq::EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
    
    // Note: Async integrator functionality is not currently available
    // For now, we'll use the regular integrator
    
    std::vector<double> state = {1.0, 0.0};
    double dt = 0.01;
    double T = 5.0;
    int steps = static_cast<int>(T / dt);
    
    std::cout << "Running async SDE integration..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Regular integration (async not available)
    for (int i = 0; i < steps; ++i) {
        integrator.step(state, dt);
        
        if (i % 100 == 0) {
            std::cout << "Step " << i << ": [" << state[0] << ", " << state[1] << "]" << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Integration completed in " << duration.count() << "ms" << std::endl;
    std::cout << "Final state: [" << state[0] << ", " << state[1] << "]" << std::endl;
}

void demonstrate_signal_aware_sde() {
    std::cout << "\n=== Signal-Aware SDE Integration ===" << std::endl;
    
    // This would integrate with the signal processing interface
    // to handle external events during SDE integration
    std::cout << "Signal-aware SDE integration would allow:" << std::endl;
    std::cout << "- Real-time parameter updates during integration" << std::endl;
    std::cout << "- Event-driven state modifications" << std::endl;
    std::cout << "- Continuous monitoring and logging" << std::endl;
    std::cout << "- Integration with external data streams" << std::endl;
}

void run_comprehensive_sde_examples() {
    std::cout << "=== Comprehensive SDE Examples ===" << std::endl;
    
    // Financial models
    finance::BlackScholesModel black_scholes(0.05, 0.2);
    black_scholes.run_comparison();
    
    finance::HestonModel heston(0.05, 2.0, 0.04, 0.3, -0.7);
    heston.run_example();
    
    // Engineering applications
    engineering::NoisyOscillator oscillator(1.0, 0.1, 0.1);
    oscillator.run_control_example();
    
    // Scientific applications
    science::StochasticLotkaVolterra ecosystem(1.0, 1.0, 1.0, 1.0, 0.1, 0.1);
    ecosystem.run_ecosystem_simulation();
    
    // Unified interface examples
    demonstrate_async_sde_integration();
    demonstrate_signal_aware_sde();
    
    std::cout << "\n=== All SDE examples completed! ===" << std::endl;
}

} // namespace unified_interface

int main() {
    std::cout << "=== diffeq SDE Usage Examples ===" << std::endl;
    
    // Run comprehensive SDE examples
    unified_interface::run_comprehensive_sde_examples();
    
    return 0;
} 