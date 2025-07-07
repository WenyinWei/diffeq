#include <diffeq.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing new SDE integrator hierarchy" << std::endl;
    
    // Simple SDE: dX = 0.05*X*dt + 0.2*X*dW (Geometric Brownian Motion)
    auto drift = [](double t, const std::vector<double>& x, std::vector<double>& fx) {
        fx[0] = 0.05 * x[0];  // 5% drift
    };
    
    auto diffusion = [](double t, const std::vector<double>& x, std::vector<double>& gx) {
        gx[0] = 0.2 * x[0];  // 20% volatility
    };
    
    // Create SDE problem and Wiener process
    auto problem = diffeq::sde::factory::make_sde_problem<std::vector<double>>(
        drift, diffusion, diffeq::sde::NoiseType::DIAGONAL_NOISE);
    auto wiener = diffeq::sde::factory::make_wiener_process<std::vector<double>>(1, 42);
    
    std::vector<double> state = {100.0};  // Initial state
    double dt = 0.01;
    double T = 1.0;
    int steps = static_cast<int>(T / dt);
    
    // Test Euler-Maruyama
    {
        std::cout << "\n=== Testing Euler-Maruyama ===" << std::endl;
        diffeq::EulerMaruyamaIntegrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    // Test Milstein
    {
        std::cout << "\n=== Testing Milstein ===" << std::endl;
        diffeq::MilsteinIntegrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    // Test SRI1
    {
        std::cout << "\n=== Testing SRI1 ===" << std::endl;
        diffeq::SRI1Integrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    // Test SRA1
    {
        std::cout << "\n=== Testing SRA1 ===" << std::endl;
        diffeq::SRA1Integrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    // Test SOSRA
    {
        std::cout << "\n=== Testing SOSRA ===" << std::endl;
        diffeq::SOSRAIntegrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    // Test SRIW1
    {
        std::cout << "\n=== Testing SRIW1 ===" << std::endl;
        diffeq::SRIW1Integrator<std::vector<double>> integrator(problem, wiener);
        std::vector<double> S = state;
        integrator.set_time(0.0);
        wiener->set_seed(12345);
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(S, dt);
        }
        std::cout << "Final value: " << S[0] << std::endl;
    }
    
    std::cout << "\nAll SDE integrators tested successfully!" << std::endl;
    return 0;
}
