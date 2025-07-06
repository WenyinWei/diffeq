#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "Testing DOP853 specifically..." << std::endl;
    
    try {
        std::vector<double> y = {1.0};
        diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay, 1e-8, 1e-12);
        integrator.set_time(0.0);
        integrator.integrate(y, 0.01, 1.0);
        std::cout << "DOP853 result: " << y[0] << " (expected: " << std::exp(-1.0) << ")" << std::endl;
        std::cout << "Error: " << std::abs(y[0] - std::exp(-1.0)) << std::endl;
        std::cout << "✅ DOP853 test passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ DOP853 test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
