#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Fixed Step Test ===" << std::endl;
    
    double t_start = 0.0, t_end = 1.0;
    std::vector<double> y0 = {1.0};
    
    // Test with fixed step sizes
    std::vector<double> step_sizes = {0.1, 0.01, 0.001};
    
    for (auto dt : step_sizes) {
        std::cout << "\nTesting with fixed dt=" << dt << std::endl;
        
        try {
            auto y = y0;
            auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
            integrator.set_time(t_start);
            
            double t = t_start;
            int step_count = 0;
            while (t < t_end) {
                double actual_dt = (dt < t_end - t) ? dt : (t_end - t);
                if (step_count < 5 || step_count % 10 == 0) {  // Reduce output
                    std::cout << "  Step " << step_count << ": t=" << t << ", y=" << y[0] << ", dt=" << actual_dt << std::endl;
                }
                integrator.step(y, actual_dt);
                integrator.set_time(t + actual_dt);  // Update integrator time
                t += actual_dt;
                step_count++;
            }
            std::cout << "  Total steps: " << step_count << std::endl;
            
            std::cout << "Success: y=" << y[0] << ", exact=" << std::exp(-1.0) << ", error=" << std::abs(y[0] - std::exp(-1.0)) << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed: " << e.what() << std::endl;
        }
    }
    
    return 0;
}
