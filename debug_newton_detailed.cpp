#include <iostream>
#include <vector>
#include <cmath>
#include "integrators/ode/bdf.hpp"

// Simple exponential decay system: dy/dt = -y
class ExponentialDecaySystem {
public:
    void operator()(double t, const std::vector<double>& y, std::vector<double>& dydt) const {
        dydt[0] = -y[0];
    }
};

int main() {
    std::cout << "=== BDF Newton Detailed Debug ===" << std::endl;
    
    ExponentialDecaySystem sys;
    
    // Test a few steps manually to see what's happening
    std::vector<double> y = {1.0};
    double dt = 0.01;
    double t = 0.0;
    
    std::cout << "Manual BDF1 Newton iteration for exponential decay" << std::endl;
    std::cout << "dt = " << dt << std::endl;
    std::cout << "Expected result: y_new = y_current / (1 + dt) = " << y[0] / (1.0 + dt) << std::endl;
    std::cout << std::endl;
    
    // Manual Newton iteration
    std::vector<double> y_current = y;
    std::vector<double> y_new = y;
    double t_new = t + dt;
    
    // Initial guess: explicit Euler
    std::vector<double> f_current(1);
    sys(t, y_current, f_current);
    y_new[0] = y_current[0] + dt * f_current[0];
    
    std::cout << "Initial guess (explicit Euler): " << y_new[0] << std::endl;
    
    // Newton iterations
    for (int iter = 0; iter < 10; ++iter) {
        // Evaluate function
        std::vector<double> f(1);
        sys(t_new, y_new, f);
        
        // Compute residual: F(y_new) = y_new - dt * f(t_new, y_new) - y_current
        double residual = y_new[0] - dt * f[0] - y_current[0];
        
        // Estimate Jacobian: for dy/dt = -y, J = -1
        double jac_diag = -1.0;  // We know this exactly for exponential decay
        
        // Newton step: solve (1 - dt*J) * dy = -residual
        double denominator = 1.0 - dt * jac_diag;
        double dy = -residual / denominator;
        
        std::cout << "Iter " << iter << ": y=" << y_new[0] << ", f=" << f[0] << ", residual=" << residual 
                  << ", jac=" << jac_diag << ", denom=" << denominator << ", dy=" << dy << std::endl;
        
        // Update
        y_new[0] += dy;
        
        // Check convergence
        if (std::abs(residual) < 1e-10) {
            std::cout << "Converged after " << iter+1 << " iterations" << std::endl;
            break;
        }
    }
    
    std::cout << "Final result: " << y_new[0] << std::endl;
    std::cout << "Expected: " << y_current[0] / (1.0 + dt) << std::endl;
    std::cout << "Error: " << std::abs(y_new[0] - y_current[0] / (1.0 + dt)) << std::endl;
    
    return 0;
}
