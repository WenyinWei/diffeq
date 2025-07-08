#include <iostream>
#include <cmath>
#include <vector>

int main() {
    std::cout << "=== BDF1 Mathematical Analysis ===" << std::endl;
    
    double t_end = 1.0;
    double y0 = 1.0;
    double exact = std::exp(-t_end);
    
    std::vector<double> step_sizes = {0.1, 0.01, 0.001, 0.0001};
    
    for (auto dt : step_sizes) {
        int n_steps = static_cast<int>(t_end / dt);
        double actual_t = n_steps * dt;
        
        // BDF1 exact formula: y_n = y0 / (1 + dt)^n
        double bdf1_result = y0 / std::pow(1.0 + dt, n_steps);
        
        // Theoretical limit as dt->0: (1 + dt)^(1/dt) -> e
        double theoretical_limit = y0 / std::pow(1.0 + dt, 1.0/dt);
        
        std::cout << "dt=" << dt << ", n_steps=" << n_steps << ", t=" << actual_t << std::endl;
        std::cout << "  BDF1 result: " << bdf1_result << std::endl;
        std::cout << "  Exact result: " << exact << std::endl;
        std::cout << "  Error: " << std::abs(bdf1_result - exact) << std::endl;
        std::cout << "  Theoretical limit: " << theoretical_limit << std::endl;
        std::cout << "  (1+dt)^(1/dt): " << std::pow(1.0 + dt, 1.0/dt) << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}
