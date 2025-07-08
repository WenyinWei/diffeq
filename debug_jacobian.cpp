#include <iostream>
#include <vector>
#include <cmath>
#include "integrators/ode/bdf.hpp"

// Simple exponential decay system: dy/dt = -y
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Jacobian Debug ===" << std::endl;
    
    // Test the Jacobian estimation in the BDF integrator
    std::vector<double> y = {1.0};
    double t = 0.0;
    
    auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
    integrator.set_time(t);
    
    // We need to access the private estimate_jacobian_diagonal method
    // Let's create a simple test to see what the integrator actually does
    
    std::cout << "Testing single step with dt=0.01" << std::endl;
    std::cout << "Initial: y=" << y[0] << std::endl;
    
    double dt = 0.01;
    integrator.step(y, dt);
    
    std::cout << "After step: y=" << y[0] << std::endl;
    std::cout << "Expected: " << 1.0 / (1.0 + dt) << std::endl;
    std::cout << "Error: " << std::abs(y[0] - 1.0 / (1.0 + dt)) << std::endl;
    
    // Test multiple steps to see where it goes wrong
    std::cout << "\nTesting multiple steps:" << std::endl;
    y = {1.0};
    integrator.set_time(0.0);
    
    for (int i = 0; i < 10; ++i) {
        double t_before = integrator.current_time();
        double y_before = y[0];
        integrator.step(y, dt);
        double t_after = integrator.current_time();
        double y_after = y[0];
        
        double expected = y_before / (1.0 + dt);
        double error = std::abs(y_after - expected);
        
        std::cout << "Step " << i << ": t=" << t_after << ", y=" << y_after 
                  << ", expected=" << expected << ", error=" << error << std::endl;
        
        if (error > 1e-10) {
            std::cout << "ERROR: Step " << i << " has significant error!" << std::endl;
            break;
        }
    }
    
    return 0;
}
