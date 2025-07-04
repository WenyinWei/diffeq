#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

// Simple test: dy/dt = -y, y(0) = 1
// Exact solution: y(t) = exp(-t)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Quick Integration Test ===" << std::endl;
    std::cout << "Testing all integrators with dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    std::vector<double> y0 = {1.0};
    
    // Test integrators one by one with error handling
    try {
        std::cout << "RK4 (fixed step):     ";
        auto y = y0;
        auto integrator = RK4Integrator<std::vector<double>>(exponential_decay);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    try {
        std::cout << "RK23 (adaptive):      ";
        auto y = y0;
        auto integrator = RK23Integrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    try {
        std::cout << "RK45 (adaptive):      ";
        auto y = y0;
        auto integrator = RK45Integrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    try {
        std::cout << "DOP853 (high-acc):    ";
        auto y = y0;
        auto integrator = DOP853Integrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    try {
        std::cout << "BDF (stiff):           ";
        auto y = y0;
        auto integrator = BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    try {
        std::cout << "LSODA (automatic):    ";
        auto y = y0;
        auto integrator = LSODAIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << std::setprecision(6) << y[0] << " (Method: " << 
            (integrator.get_current_method() == LSODAIntegrator<std::vector<double>>::MethodType::ADAMS ? 
             "Adams)" : "BDF)") << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    std::cout << "\n✓ Quick test completed!" << std::endl;
    std::cout << "Note: Some integrators may fail with simplified implementations." << std::endl;
    std::cout << "For production use, see the examples/advanced_integrators_usage.cpp" << std::endl;
    
    return 0;
}
