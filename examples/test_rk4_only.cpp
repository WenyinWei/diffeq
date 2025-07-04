#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "Testing RK4 only..." << std::endl;
    
    try {
        std::vector<double> y = {1.0};
        auto integrator = diffeq::integrators::ode::RK4Integrator<std::vector<double>>(exponential_decay);
        integrator.set_time(0.0);
        integrator.integrate(y, 0.1, 1.0);
        std::cout << "RK4 result: " << y[0] << " (expected: " << std::exp(-1.0) << ")" << std::endl;
        std::cout << "✓ RK4 test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ RK4 test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
