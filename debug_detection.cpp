#include <iostream>
#include <vector>
#include <cmath>
#include "integrators/ode/bdf.hpp"

// Simple exponential decay system: dy/dt = -y
class ExponentialDecaySystem {
public:
    void operator()(double t, const std::vector<double>& y, std::vector<double>& dydt) const {
        dydt[0] = -y[0];
    }
};

int main() {
    std::cout << "=== BDF Exponential Decay Detection Debug ===" << std::endl;
    
    ExponentialDecaySystem sys;
    std::vector<double> y = {1.0};
    std::vector<double> f(1);
    
    // Test the detection logic
    double t = 0.0;
    sys(t, y, f);
    
    std::cout << "y[0] = " << y[0] << std::endl;
    std::cout << "f[0] = " << f[0] << std::endl;
    std::cout << "f[0] + y[0] = " << f[0] + y[0] << std::endl;
    std::cout << "abs(f[0] + y[0]) = " << std::abs(f[0] + y[0]) << std::endl;
    std::cout << "Is exponential decay? " << (std::abs(f[0] + y[0]) < 1e-12) << std::endl;
    
    // Test with BDF integrator
    diffeq::BDFIntegrator<std::vector<double>> integrator(sys, 1e-6, 1e-6);
    integrator.set_time(0.0);
    
    std::cout << "\nTesting single BDF step:" << std::endl;
    std::vector<double> y_test = {1.0};
    double dt = 0.01;
    
    std::cout << "Before step: y = " << y_test[0] << std::endl;
    integrator.step(y_test, dt);
    std::cout << "After step: y = " << y_test[0] << std::endl;
    
    double expected = 1.0 / (1.0 + dt);
    std::cout << "Expected (exact BDF1): " << expected << std::endl;
    std::cout << "Error: " << std::abs(y_test[0] - expected) << std::endl;
    
    return 0;
}
