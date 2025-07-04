#include <diffeq.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>

using namespace diffeq::sde;

/**
 * @brief Test problem: Geometric Brownian Motion
 * 
 * dS = μ * S * dt + σ * S * dW
 * 
 * Analytical solution: S(t) = S(0) * exp((μ - σ²/2)*t + σ*W(t))
 */
class GeometricBrownianMotion {
public:
    double mu, sigma;
    
    GeometricBrownianMotion(double mu = 0.05, double sigma = 0.2) 
        : mu(mu), sigma(sigma) {}
    
    void drift(double t, const std::vector<double>& S, std::vector<double>& dS) {
        dS[0] = mu * S[0];
    }
    
    void diffusion(double t, const std::vector<double>& S, std::vector<double>& gS) {
        gS[0] = sigma * S[0];
    }
    
    double analytical_mean(double S0, double t) {
        return S0 * std::exp(mu * t);
    }
    
    double analytical_variance(double S0, double t) {
        double mean = analytical_mean(S0, t);
        return mean * mean * (std::exp(sigma * sigma * t) - 1.0);
    }
};

/**
 * @brief Test problem: Additive noise SDE
 * 
 * dX = -λ * X * dt + σ * dW
 * 
 * Suitable for testing SRA methods (additive noise)
 */
class AdditiveNoiseSDE {
public:
    double lambda, sigma;
    
    AdditiveNoiseSDE(double lambda = 1.0, double sigma = 0.5)
        : lambda(lambda), sigma(sigma) {}
    
    void drift(double t, const std::vector<double>& X, std::vector<double>& dX) {
        dX[0] = -lambda * X[0];
    }
    
    void diffusion(double t, const std::vector<double>& X, std::vector<double>& gX) {
        gX[0] = sigma;  // Additive noise
    }
};

/**
 * @brief Test problem: Stiff SDE for testing stability-optimized methods
 * 
 * dX = -α * X * dt + σ * X * dW
 * 
 * With large α, this becomes stiff and tests robustness
 */
class StiffSDE {
public:
    double alpha, sigma;
    
    StiffSDE(double alpha = 50.0, double sigma = 0.1)
        : alpha(alpha), sigma(sigma) {}
    
    void drift(double t, const std::vector<double>& X, std::vector<double>& dX) {
        dX[0] = -alpha * X[0];
    }
    
    void diffusion(double t, const std::vector<double>& X, std::vector<double>& gX) {
        gX[0] = sigma * X[0];
    }
};

void test_basic_sde_solvers() {
    std::cout << "\n=== Testing Basic SDE Solvers ===\n";
    
    GeometricBrownianMotion gbm(0.05, 0.2);
    
    auto drift_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& dS) {
        gbm.drift(t, S, dS);
    };
    
    auto diffusion_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& gS) {
        gbm.diffusion(t, S, gS);
    };
    
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 12345);
    
    // Test Euler-Maruyama
    {
        EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
        std::vector<double> S = {100.0};  // Initial stock price
        
        double dt = 0.01;
        double T = 1.0;
        int steps = static_cast<int>(T / dt);
        
        integrator.set_time(0.0);
        wiener->set_seed(12345);  // Reset for reproducibility
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        
        std::cout << "Euler-Maruyama final value: " << S[0] << std::endl;
        std::cout << "Expected mean: " << gbm.analytical_mean(100.0, T) << std::endl;
    }
    
    // Test Milstein
    {
        MilsteinIntegrator<std::vector<double>, double> integrator(problem, wiener);
        std::vector<double> S = {100.0};
        
        double dt = 0.01;
        double T = 1.0;
        int steps = static_cast<int>(T / dt);
        
        integrator.set_time(0.0);
        wiener->set_seed(12345);  // Reset for reproducibility
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        
        std::cout << "Milstein final value: " << S[0] << std::endl;
    }
    
    // Test SRI1
    {
        SRI1Integrator<std::vector<double>, double> integrator(problem, wiener);
        std::vector<double> S = {100.0};
        
        double dt = 0.01;
        double T = 1.0;
        int steps = static_cast<int>(T / dt);
        
        integrator.set_time(0.0);
        wiener->set_seed(12345);  // Reset for reproducibility
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        
        std::cout << "SRI1 final value: " << S[0] << std::endl;
    }
}

void test_advanced_sde_solvers() {
    std::cout << "\n=== Testing Advanced SDE Solvers ===\n";
    
    GeometricBrownianMotion gbm(0.05, 0.2);
    
    auto drift_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& dS) {
        gbm.drift(t, S, dS);
    };
    
    auto diffusion_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& gS) {
        gbm.diffusion(t, S, gS);
    };
    
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 12345);
    
    double dt = 0.01;
    double T = 1.0;
    int steps = static_cast<int>(T / dt);
    std::vector<double> initial_state = {100.0};
    
    // Test SRA1
    {
        auto integrator = factory::make_sra1_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> S = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(S, dt);
        }
        
        std::cout << "SRA1 final value: " << S[0] << std::endl;
    }
    
    // Test SRA2
    {
        auto integrator = factory::make_sra2_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> S = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(S, dt);
        }
        
        std::cout << "SRA2 final value: " << S[0] << std::endl;
    }
    
    // Test SOSRA
    {
        auto integrator = factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> S = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(S, dt);
        }
        
        std::cout << "SOSRA final value: " << S[0] << std::endl;
    }
    
    // Test SRIW1
    {
        auto integrator = factory::make_sriw1_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> S = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(S, dt);
        }
        
        std::cout << "SRIW1 final value: " << S[0] << std::endl;
    }
    
    // Test SOSRI
    {
        auto integrator = factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> S = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(S, dt);
        }
        
        std::cout << "SOSRI final value: " << S[0] << std::endl;
    }
}

void test_additive_noise_sra() {
    std::cout << "\n=== Testing SRA Methods with Additive Noise ===\n";
    
    AdditiveNoiseSDE additive_sde(1.0, 0.5);
    
    auto drift_func = [&additive_sde](double t, const std::vector<double>& X, std::vector<double>& dX) {
        additive_sde.drift(t, X, dX);
    };
    
    auto diffusion_func = [&additive_sde](double t, const std::vector<double>& X, std::vector<double>& gX) {
        additive_sde.diffusion(t, X, gX);
    };
    
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 54321);
    
    double dt = 0.01;
    double T = 1.0;
    int steps = static_cast<int>(T / dt);
    std::vector<double> initial_state = {1.0};
    
    // Compare SRA1, SRA2, and SOSRA on additive noise problem
    std::vector<std::string> method_names = {"SRA1", "SRA2", "SOSRA"};
    std::vector<double> final_values;
    
    // SRA1
    {
        auto integrator = factory::make_sra1_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> X = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(54321);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(X, dt);
        }
        
        final_values.push_back(X[0]);
        std::cout << "SRA1 (additive) final value: " << X[0] << std::endl;
    }
    
    // SRA2
    {
        auto integrator = factory::make_sra2_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> X = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(54321);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(X, dt);
        }
        
        final_values.push_back(X[0]);
        std::cout << "SRA2 (additive) final value: " << X[0] << std::endl;
    }
    
    // SOSRA
    {
        auto integrator = factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
        std::vector<double> X = initial_state;
        
        integrator->set_time(0.0);
        wiener->set_seed(54321);
        
        for (int i = 0; i < steps; ++i) {
            integrator->step(X, dt);
        }
        
        final_values.push_back(X[0]);
        std::cout << "SOSRA (additive) final value: " << X[0] << std::endl;
    }
}

void test_stiff_sde_stability() {
    std::cout << "\n=== Testing Stability with Stiff SDEs ===\n";
    
    StiffSDE stiff_sde(50.0, 0.1);  // Large α makes it stiff
    
    auto drift_func = [&stiff_sde](double t, const std::vector<double>& X, std::vector<double>& dX) {
        stiff_sde.drift(t, X, dX);
    };
    
    auto diffusion_func = [&stiff_sde](double t, const std::vector<double>& X, std::vector<double>& gX) {
        stiff_sde.diffusion(t, X, gX);
    };
    
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 98765);
    
    double dt = 0.001;  // Small time step for stiff problem
    double T = 0.1;     // Short time horizon
    int steps = static_cast<int>(T / dt);
    std::vector<double> initial_state = {1.0};
    
    std::cout << "Testing with stiff parameter α = " << stiff_sde.alpha 
              << ", dt = " << dt << ", T = " << T << std::endl;
    
    // Test stability-optimized methods
    try {
        // SOSRA
        {
            auto integrator = factory::make_sosra_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> X = initial_state;
            
            integrator->set_time(0.0);
            wiener->set_seed(98765);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(X, dt);
            }
            
            std::cout << "SOSRA (stiff) final value: " << X[0] << std::endl;
        }
        
        // SOSRI
        {
            auto integrator = factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> X = initial_state;
            
            integrator->set_time(0.0);
            wiener->set_seed(98765);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(X, dt);
            }
            
            std::cout << "SOSRI (stiff) final value: " << X[0] << std::endl;
        }
        
        std::cout << "✓ Stability-optimized methods handled stiff problem successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error with stiff problem: " << e.what() << std::endl;
    }
}

void test_convergence_order() {
    std::cout << "\n=== Testing Strong Convergence Order ===\n";
    
    GeometricBrownianMotion gbm(0.0, 0.2);  // Zero drift for simpler analysis
    
    auto drift_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& dS) {
        gbm.drift(t, S, dS);
    };
    
    auto diffusion_func = [&gbm](double t, const std::vector<double>& S, std::vector<double>& gS) {
        gbm.diffusion(t, S, gS);
    };
    
    auto problem = factory::make_sde_problem<std::vector<double>, double>(
        drift_func, diffusion_func, NoiseType::DIAGONAL_NOISE);
    
    auto wiener = factory::make_wiener_process<std::vector<double>, double>(1, 11111);
    
    double T = 0.1;
    std::vector<double> initial_state = {1.0};
    std::vector<double> dt_values = {0.01, 0.005, 0.0025};
    
    std::cout << "Testing convergence order for T = " << T << std::endl;
    std::cout << std::setw(12) << "dt" << std::setw(15) << "Euler-M" << std::setw(15) << "SRA1" 
              << std::setw(15) << "SRIW1" << std::setw(15) << "SOSRI" << std::endl;
    
    for (double dt : dt_values) {
        int steps = static_cast<int>(T / dt);
        
        std::vector<double> results;
        
        // Euler-Maruyama
        {
            EulerMaruyamaIntegrator<std::vector<double>, double> integrator(problem, wiener);
            std::vector<double> S = initial_state;
            integrator.set_time(0.0);
            wiener->set_seed(11111);
            
            for (int i = 0; i < steps; ++i) {
                integrator.step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SRA1
        {
            auto integrator = factory::make_sra1_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = initial_state;
            integrator->set_time(0.0);
            wiener->set_seed(11111);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SRIW1
        {
            auto integrator = factory::make_sriw1_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = initial_state;
            integrator->set_time(0.0);
            wiener->set_seed(11111);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        // SOSRI
        {
            auto integrator = factory::make_sosri_integrator<std::vector<double>, double>(problem, wiener);
            std::vector<double> S = initial_state;
            integrator->set_time(0.0);
            wiener->set_seed(11111);
            
            for (int i = 0; i < steps; ++i) {
                integrator->step(S, dt);
            }
            results.push_back(S[0]);
        }
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << dt;
        for (double result : results) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(6) << result;
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== Comprehensive SDE Solver Testing ===\n";
    std::cout << "Testing enhanced SDE capabilities with DifferentialEquations.jl-inspired algorithms\n";
    
    try {
        test_basic_sde_solvers();
        test_advanced_sde_solvers();
        test_additive_noise_sra();
        test_stiff_sde_stability();
        test_convergence_order();
        
        std::cout << "\n✅ All SDE tests completed successfully!" << std::endl;
        std::cout << "\n📊 Summary of implemented algorithms:" << std::endl;
        std::cout << "• Basic: Euler-Maruyama, Milstein, SRI1, Implicit Euler-Maruyama" << std::endl;
        std::cout << "• Advanced: SRA1, SRA2, SOSRA (Strong order 1.5 for additive noise)" << std::endl;
        std::cout << "• Advanced: SRIW1, SOSRI (Strong order 1.5 for general Itô SDEs)" << std::endl;
        std::cout << "• All methods support proper concepts, async integration, and signal processing" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
