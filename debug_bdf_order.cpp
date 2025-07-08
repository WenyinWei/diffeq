#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <integrators/ode/bdf_scipy.hpp>

// Simple exponential decay: dy/dt = -y
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Order Debug ===" << std::endl;
    std::cout << "Testing dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;

    std::vector<double> y = {1.0};
    double t_start = 0.0;
    double t_end = 1.0;
    double dt = 0.1;

    auto integrator = diffeq::ScipyBDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
    integrator.set_time(t_start);

    std::cout << "Integration steps:" << std::endl;
    std::cout << "Step | Time     | y        | Order | h" << std::endl;
    std::cout << "-----|----------|----------|-------|----------" << std::endl;

    int step = 0;
    double current_time = t_start;
    
    while (current_time < t_end) {
        double h = integrator.adaptive_step(y, dt);
        current_time = integrator.get_current_time();
        step++;
        
        std::cout << std::setw(4) << step << " | "
                  << std::setw(8) << std::setprecision(5) << current_time << " | "
                  << std::setw(8) << std::setprecision(6) << y[0] << " | "
                  << std::setw(5) << integrator.get_current_order() << " | "
                  << std::setw(8) << std::setprecision(5) << h << std::endl;
        
        if (step > 20) break; // Prevent infinite loops
    }

    std::cout << std::endl;
    std::cout << "Final result: " << std::setprecision(6) << y[0] << std::endl;
    std::cout << "Expected:     " << std::setprecision(6) << std::exp(-1.0) << std::endl;
    std::cout << "Error:        " << std::setprecision(6) << std::abs(y[0] - std::exp(-1.0)) << std::endl;

    return 0;
}
