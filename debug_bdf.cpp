#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Debug Test ===" << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    std::vector<double> y0 = {1.0};
    
    // Test with different tolerances - start with very loose tolerances
    std::vector<std::pair<double, double>> tolerances = {
        {1e-1, 1e-3},   // Very loose tolerances
        {1e-2, 1e-4},   // Loose tolerances
        {1e-3, 1e-6}    // Original tolerances
    };
    
    for (auto& tol : tolerances) {
        std::cout << "\nTesting with rtol=" << tol.first << ", atol=" << tol.second << std::endl;
        
        try {
            auto y = y0;
            auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, tol.first, tol.second);
            integrator.set_time(t_start);

            // Set more reasonable step limits
            integrator.set_step_limits(1e-10, 1.0);
            
            integrator.integrate(y, dt, t_end);
            std::cout << "Success: y=" << y[0] << ", exact=" << std::exp(-1.0) << ", error=" << std::abs(y[0] - std::exp(-1.0)) << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed: " << e.what() << std::endl;
        }
    }
    
    return 0;
}
