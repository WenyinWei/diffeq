#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== BDF Adaptive Step Debug ===" << std::endl;
    
    std::vector<double> y = {1.0};
    auto integrator = diffeq::BDFIntegrator<std::vector<double>>(exponential_decay, 1e-3, 1e-6);
    integrator.set_time(0.0);
    
    double dt = 0.1;
    double t_end = 1.0;
    
    std::cout << "Initial: t=" << integrator.current_time() << ", y=" << y[0] << std::endl;
    std::cout << "Target: integrate from t=0 to t=" << t_end << " with dt=" << dt << std::endl;
    std::cout << "Expected final result: " << std::exp(-t_end) << std::endl;
    std::cout << std::endl;
    
    // Manual stepping to see what happens
    int step_count = 0;
    const int max_steps = 50;  // Prevent infinite loops
    
    while (integrator.current_time() < t_end && step_count < max_steps) {
        double t_before = integrator.current_time();
        double y_before = y[0];
        
        std::cout << "=== Step " << step_count << " ===" << std::endl;
        std::cout << "Before: t=" << std::fixed << std::setprecision(6) << t_before 
                  << ", y=" << y_before << std::endl;
        
        // Calculate the step size we want to take
        double remaining_time = t_end - t_before;
        double step_size = (dt < remaining_time) ? dt : remaining_time;
        
        std::cout << "Requested step size: " << step_size << std::endl;
        
        try {
            // Take one step
            integrator.step(y, step_size);
            
            double t_after = integrator.current_time();
            double y_after = y[0];
            double actual_step = t_after - t_before;
            
            std::cout << "After:  t=" << std::fixed << std::setprecision(6) << t_after 
                      << ", y=" << y_after << std::endl;
            std::cout << "Actual step size: " << actual_step << std::endl;
            
            // Check if we made progress
            if (actual_step <= 0) {
                std::cout << "ERROR: No progress made (step size <= 0)" << std::endl;
                break;
            }
            
            if (actual_step < step_size * 0.1) {
                std::cout << "WARNING: Step size much smaller than requested" << std::endl;
            }
            
            // Calculate expected value and error
            double expected = std::exp(-t_after);
            double error = std::abs(y_after - expected);
            std::cout << "Expected: " << expected << ", Error: " << error << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "EXCEPTION: " << e.what() << std::endl;
            break;
        }
        
        step_count++;
        std::cout << std::endl;
        
        // Safety check for very small steps
        if (integrator.current_time() - t_before < 1e-12) {
            std::cout << "ERROR: Step size became too small" << std::endl;
            break;
        }
    }
    
    if (step_count >= max_steps) {
        std::cout << "ERROR: Maximum steps reached" << std::endl;
    }
    
    std::cout << "=== Final Result ===" << std::endl;
    std::cout << "Steps taken: " << step_count << std::endl;
    std::cout << "Final time: " << integrator.current_time() << std::endl;
    std::cout << "Final y: " << y[0] << std::endl;
    std::cout << "Expected: " << std::exp(-integrator.current_time()) << std::endl;
    std::cout << "Error: " << std::abs(y[0] - std::exp(-integrator.current_time())) << std::endl;
    
    return 0;
}
