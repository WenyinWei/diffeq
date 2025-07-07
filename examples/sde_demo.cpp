#include <diffeq.hpp>
#include <iostream>

int main() {
    try {
        std::cout << "=== Basic SDE Demo ===" << std::endl;
        
        // Simple SDE: dX = -0.1*X*dt + 0.1*X*dW (Geometric Brownian Motion)
        auto drift_func = [](double t, const std::vector<double>& x, std::vector<double>& dx) {
            dx[0] = -0.1 * x[0];
        };
        
        auto diffusion_func = [](double t, const std::vector<double>& x, std::vector<double>& gx) {
            gx[0] = 0.1 * x[0];
        };
        
        // Create SDE problem and Wiener process directly without factory functions
        auto problem = std::make_shared<diffeq::sde::SDEProblem<std::vector<double>>>(
            drift_func, diffusion_func, diffeq::sde::NoiseType::DIAGONAL_NOISE);
        
        auto wiener = std::make_shared<diffeq::sde::WienerProcess<std::vector<double>>>(1, 12345);
        
        diffeq::EulerMaruyamaIntegrator<std::vector<double>> integrator(problem, wiener);
        
        std::vector<double> state = {1.0};  // Initial condition
        double dt = 0.01;
        double T = 1.0;
        int steps = static_cast<int>(T / dt);
        
        std::cout << "Initial state: " << state[0] << std::endl;
        
        for (int i = 0; i < steps; ++i) {
            integrator.step(state, dt);
            if (i % 25 == 0) {  // Output every quarter
                std::cout << "t = " << (i * dt) << ": X = " << state[0] << std::endl;
            }
        }
        
        std::cout << "Final state: " << state[0] << std::endl;
        std::cout << "SDE integration completed successfully!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
