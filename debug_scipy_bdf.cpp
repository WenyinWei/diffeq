#include <diffeq.hpp>
#include <integrators/ode/simple_bdf1.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

// Simple test: dy/dt = -y, y(0) = 1
// Exact solution: y(t) = exp(-t)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Debug SciPy BDF Implementation ===" << std::endl;
    std::cout << "Testing dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    std::vector<double> y0 = {1.0};
    
    // Test simple BDF1 first
    try {
        std::cout << "Testing SimpleBDF1Integrator..." << std::endl;
        auto y = y0;
        auto integrator = diffeq::SimpleBDF1Integrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);

        // Take a few manual steps to debug
        std::cout << "Initial: t=" << integrator.current_time() << ", y=" << y[0] << std::endl;

        for (int i = 0; i < 5; ++i) {
            double h = integrator.adaptive_step(y, 0.2);
            std::cout << "Step " << i+1 << ": t=" << integrator.current_time()
                      << ", y=" << y[0] << ", h=" << h << std::endl;
        }

        std::cout << "\nSimple BDF1 result: " << y[0] << std::endl;
        std::cout << "Expected: " << std::exp(-integrator.current_time()) << std::endl;

    } catch (const std::exception& e) {
        std::cout << "SimpleBDF1 failed: " << e.what() << std::endl;
    }

    std::cout << "\n" << std::endl;

    try {
        std::cout << "Testing ScipyBDFIntegrator..." << std::endl;
        auto y = y0;
        auto integrator = diffeq::ScipyBDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);

        // Let's manually check what the BDF coefficients should be
        std::cout << "Expected BDF1 coefficients:" << std::endl;
        std::cout << "gamma[1] = 1.0, alpha[1] = 1.0, error_const[1] = 0.5" << std::endl;
        
        // Take a few manual steps to debug
        std::cout << "Initial: t=" << integrator.current_time() << ", y=" << y[0] << std::endl;

        for (int i = 0; i < 3; ++i) {
            double h = integrator.adaptive_step(y, 0.2);
            std::cout << "Step " << i+1 << ": t=" << integrator.current_time()
                      << ", y=" << y[0] << ", h=" << h << std::endl;
        }

        // Let's also test with a larger initial step to see if it behaves better
        std::cout << "\nTesting with larger initial step (0.05):" << std::endl;
        auto y2 = y0;
        auto integrator2 = diffeq::ScipyBDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator2.set_time(t_start);

        for (int i = 0; i < 3; ++i) {
            double h = integrator2.adaptive_step(y2, 0.05);
            std::cout << "Step " << i+1 << ": t=" << integrator2.current_time()
                      << ", y=" << y2[0] << ", h=" << h << std::endl;
        }

        std::cout << "\nSciPy BDF result: " << y[0] << std::endl;
        std::cout << "Expected: " << std::exp(-integrator.current_time()) << std::endl;

    } catch (const std::exception& e) {
        std::cout << "SciPy BDF failed: " << e.what() << std::endl;
    }
    
    return 0;
}
