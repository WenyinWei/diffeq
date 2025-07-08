#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "=== Simple BDF1 (Backward Euler) Test ===" << std::endl;
    std::cout << "Testing dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Analytical solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    double y = 1.0;
    double t = 0.0;
    double t_end = 1.0;
    double h = 0.01;  // Small step size
    
    std::cout << "Using backward Euler: y_{n+1} = y_n / (1 + h)" << std::endl;
    std::cout << "Step size h = " << h << std::endl << std::endl;
    
    int step = 0;
    while (t < t_end) {
        double h_actual = (t + h > t_end) ? (t_end - t) : h;
        
        // Backward Euler for dy/dt = -y
        // y_{n+1} = y_n / (1 + h)
        y = y / (1.0 + h_actual);
        t += h_actual;
        
        if (step % 10 == 0 || t >= t_end) {
            std::cout << "Step " << step << ": t=" << t << ", y=" << y 
                      << ", exact=" << std::exp(-t) 
                      << ", error=" << std::abs(y - std::exp(-t)) << std::endl;
        }
        step++;
    }
    
    std::cout << std::endl;
    std::cout << "Final result: " << std::setprecision(6) << y << std::endl;
    std::cout << "Expected:     " << std::setprecision(6) << std::exp(-1.0) << std::endl;
    std::cout << "Error:        " << std::setprecision(6) << std::abs(y - std::exp(-1.0)) << std::endl;
    
    return 0;
}
