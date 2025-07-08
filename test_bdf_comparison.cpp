#include <diffeq.hpp>
#include <integrators/ode/bdf_scipy.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

// Simple test: dy/dt = -y, y(0) = 1
// Exact solution: y(t) = exp(-t)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Implementation Comparison ===" << std::endl;
    std::cout << "Testing dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.01;  // Smaller step size
    std::vector<double> y0 = {1.0};
    
    // Test main BDFIntegrator
    try {
        std::cout << "Main BDFIntegrator:    ";
        auto y = y0;
        auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    // Test ScipyBDFIntegrator
    try {
        std::cout << "ScipyBDFIntegrator:    ";
        auto y = y0;
        auto integrator = diffeq::ScipyBDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    return 0;
}
