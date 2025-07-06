#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <memory>
#include <diffeq.hpp>

// Example 1: Simple exponential decay
// dy/dt = -k*y, where k is the decay constant
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double k = 0.5; // decay constant
    dydt[0] = -k * y[0];
}

// Example 2: Lorenz attractor (simplified)
// dx/dt = σ(y - x)
// dy/dt = x(ρ - z) - y  
// dz/dt = xy - βz
void lorenz_system(double t, const std::vector<double>& state, std::vector<double>& dydt) {
    const double sigma = 10.0;
    const double rho = 28.0;
    const double beta = 8.0/3.0;
    
    double x = state[0];
    double y = state[1]; 
    double z = state[2];
    
    dydt[0] = sigma * (y - x);
    dydt[1] = x * (rho - z) - y;
    dydt[2] = x * y - beta * z;
}

// Example 3: Damped harmonic oscillator
// d²x/dt² + 2γ(dx/dt) + ω²x = 0
// State: [x, dx/dt]
void damped_oscillator(float t, const std::array<float, 2>& state, std::array<float, 2>& dydt) {
    const float gamma = 0.1f; // damping coefficient
    const float omega_sq = 4.0f; // ω² = 4
    
    dydt[0] = state[1]; // dx/dt = v
    dydt[1] = -2.0f * gamma * state[1] - omega_sq * state[0]; // dv/dt = -2γv - ω²x
}

int main() {
    std::cout << "RK4 Integrator Usage Examples" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    // Example 1: Exponential Decay
    std::cout << "\n1. Exponential Decay (dy/dt = -0.5*y)" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    diffeq::RK4Integrator<std::vector<double>, double> decay_integrator(exponential_decay);
    std::vector<double> decay_state = {2.0}; // y(0) = 2
    
    std::cout << "Time\tValue" << std::endl;
    std::cout << "0.0\t" << decay_state[0] << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        decay_integrator.step(decay_state, 0.5);
        std::cout << decay_integrator.current_time() << "\t" << decay_state[0] << std::endl;
    }
    
    // Example 2: Lorenz System
    std::cout << "\n2. Lorenz Attractor (first 20 steps)" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    diffeq::RK4Integrator<std::vector<double>, double> lorenz_integrator(lorenz_system);
    std::vector<double> lorenz_state = {1.0, 1.0, 1.0}; // Initial conditions
    
    std::cout << "Time\tX\t\tY\t\tZ" << std::endl;
    std::cout << "0.0\t" << lorenz_state[0] << "\t\t" << lorenz_state[1] << "\t\t" << lorenz_state[2] << std::endl;
    
    for (int i = 0; i < 20; ++i) {
        lorenz_integrator.step(lorenz_state, 0.01);
        if (i % 5 == 4) { // Print every 5th step
            std::cout << lorenz_integrator.current_time() << "\t" 
                      << lorenz_state[0] << "\t\t" 
                      << lorenz_state[1] << "\t\t" 
                      << lorenz_state[2] << std::endl;
        }
    }
    
    // Example 3: Damped Harmonic Oscillator with float precision
    std::cout << "\n3. Damped Harmonic Oscillator (float precision)" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    
    diffeq::RK4Integrator<std::array<float, 2>, float> oscillator_integrator(damped_oscillator);
    std::array<float, 2> oscillator_state = {1.0f, 0.0f}; // x(0) = 1, v(0) = 0
    
    std::cout << "Time\tPosition\tVelocity" << std::endl;
    std::cout << "0.0\t" << oscillator_state[0] << "\t\t" << oscillator_state[1] << std::endl;
    
    for (int i = 0; i < 50; ++i) {
        oscillator_integrator.step(oscillator_state, 0.1f);
        if (i % 10 == 9) { // Print every 10th step
            std::cout << oscillator_integrator.current_time() << "\t" 
                      << oscillator_state[0] << "\t\t" 
                      << oscillator_state[1] << std::endl;
        }
    }
    
    // Example 4: Using polymorphism
    std::cout << "\n4. Polymorphic Usage" << std::endl;
    std::cout << "-------------------" << std::endl;
    
    auto integrator = std::make_unique<diffeq::RK4Integrator<std::vector<double>, double>>(exponential_decay);
    AbstractIntegrator<std::vector<double>, double>* base_ptr = integrator.get();
    
    std::vector<double> poly_state = {5.0};
    std::cout << "Initial: t=" << base_ptr->current_time() << ", y=" << poly_state[0] << std::endl;
    
    base_ptr->integrate(poly_state, 0.1, 2.0); // Integrate from t=0 to t=2 with dt=0.1
    std::cout << "Final: t=" << base_ptr->current_time() << ", y=" << poly_state[0] << std::endl;
    
    return 0;
}
