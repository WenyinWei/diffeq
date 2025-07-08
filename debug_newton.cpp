#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
    std::cout << "f(" << t << ", " << y[0] << ") = " << dydt[0] << std::endl;
}

int main() {
    std::cout << "=== BDF Newton Debug Test ===" << std::endl;
    
    double t_start = 0.0;
    std::vector<double> y0 = {1.0};
    double dt = 0.1;
    
    std::cout << "Initial: t=" << t_start << ", y=" << y0[0] << std::endl;
    
    try {
        auto y = y0;
        auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
        integrator.set_time(t_start);
        
        // Take one step and see what happens
        std::cout << "\nTaking one step with dt=" << dt << std::endl;
        integrator.step(y, dt);
        
        std::cout << "After step: y=" << y[0] << ", exact=" << std::exp(-dt) << ", error=" << std::abs(y[0] - std::exp(-dt)) << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << std::endl;
    }
    
    return 0;
}
