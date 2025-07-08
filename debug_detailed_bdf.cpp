#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <cmath>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Detailed BDF Debug ===" << std::endl;
    
    std::vector<double> y0 = {1.0};
    double t_start = 0.0;
    
    std::cout << "Testing dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl;
    
    try {
        auto y = y0;
        auto integrator = diffeq::ScipyBDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        
        std::cout << "\nTesting one step manually..." << std::endl;
        std::cout << "Initial: t=" << integrator.current_time() << ", y=" << y[0] << std::endl;
        
        // Let's manually test what happens with a reasonable step size
        double h_test = 0.01;  // Try a step size similar to SimpleBDF1
        std::cout << "Attempting step with h=" << h_test << std::endl;
        
        // For BDF1 with dy/dt = -y:
        // y_new = y_old / (1 + h)
        double y_expected = y[0] / (1.0 + h_test);
        std::cout << "Expected BDF1 result: " << y_expected << std::endl;
        
        // Now test the actual integrator
        double h_actual = integrator.adaptive_step(y, h_test);
        std::cout << "Actual step taken: h=" << h_actual << std::endl;
        std::cout << "Actual result: y=" << y[0] << std::endl;
        std::cout << "Difference from expected: " << std::abs(y[0] - y_expected) << std::endl;
        
        // Calculate what the error should be
        double rtol = 1e-3, atol = 1e-6;
        double scale = atol + rtol * std::abs(y[0]);
        std::cout << "Scale factor: " << scale << std::endl;
        
        // For BDF1, the local truncation error is approximately h^2/2 * y''
        // For dy/dt = -y, y'' = y, so LTE ≈ h^2/2 * y
        double lte_estimate = h_test * h_test / 2.0 * y[0];
        std::cout << "Theoretical LTE: " << lte_estimate << std::endl;
        std::cout << "Scaled LTE: " << lte_estimate / scale << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    return 0;
}
