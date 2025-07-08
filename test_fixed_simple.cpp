#include <diffeq.hpp>
#include <iostream>
#include <vector>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Simple Fixed Step Test ===" << std::endl;
    
    std::vector<double> y = {1.0};
    auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
    integrator.set_time(0.0);
    
    double dt = 0.01;
    
    std::cout << "Testing 100 steps with dt=" << dt << " to reach t=1.0" << std::endl;

    for (int i = 0; i < 100; ++i) {
        double t_before = integrator.current_time();
        double y_before = y[0];
        
        integrator.step(y, dt);
        
        double t_after = integrator.current_time();
        double y_after = y[0];
        double expected = std::exp(-t_after);
        double error = std::abs(y_after - expected);
        
        if (i < 5 || i % 10 == 9) {  // Print first 5 and every 10th step
            std::cout << "Step " << i << ": t=" << t_after << ", y=" << y_after
                      << ", expected=" << expected << ", error=" << error << std::endl;
        }
    }
    
    // Final comparison
    double final_time = integrator.current_time();
    double final_y = y[0];
    double exact_final = std::exp(-final_time);
    double final_error = std::abs(final_y - exact_final);
    
    std::cout << "\nFinal result:" << std::endl;
    std::cout << "Time: " << final_time << std::endl;
    std::cout << "Numerical: " << final_y << std::endl;
    std::cout << "Exact: " << exact_final << std::endl;
    std::cout << "Error: " << final_error << std::endl;
    
    if (final_error < 0.01) {  // More reasonable tolerance for BDF1
        std::cout << "SUCCESS: Error is within tolerance" << std::endl;
    } else {
        std::cout << "FAILURE: Error is too large" << std::endl;
    }
    
    return 0;
}
