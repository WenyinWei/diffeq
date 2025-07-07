#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <integrators/ode/rk4.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

// Simple test: dy/dt = -y, y(0) = 1
// Exact solution: y(t) = exp(-t)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Simple RK4 Test ===" << std::endl;
    std::cout << "Testing basic RK4 with dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    std::vector<double> y0 = {1.0};
    
    try {
        std::cout << "RK4 (fixed step):     ";
        auto y = y0;
        auto integrator = integrators::ode::RK4Integrator<std::vector<double>>(exponential_decay);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
        std::cout << "✓ Basic RK4 test successful!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 