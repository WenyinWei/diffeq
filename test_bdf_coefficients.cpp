#include <iostream>
#include <array>
#include <iomanip>

int main() {
    std::cout << "=== BDF Coefficients Test ===" << std::endl;
    
    const int MAX_ORDER = 5;
    
    // Standard BDF coefficients (no kappa)
    // gamma[k] = sum(1/j for j=1..k)
    std::array<double, MAX_ORDER + 2> gamma;
    gamma[0] = 0.0;
    for (int k = 1; k <= MAX_ORDER; ++k) {
        gamma[k] = gamma[k-1] + 1.0 / k;
    }

    // alpha = gamma (standard BDF)
    std::array<double, MAX_ORDER + 2> alpha;
    for (int k = 0; k <= MAX_ORDER; ++k) {
        alpha[k] = gamma[k];
    }

    // error_const = 1/(k+1) (standard BDF)
    std::array<double, MAX_ORDER + 2> error_const;
    for (int k = 0; k <= MAX_ORDER; ++k) {
        error_const[k] = 1.0 / (k + 1);
    }
    
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "\nCalculated coefficients:" << std::endl;
    for (int k = 1; k <= MAX_ORDER; ++k) {
        std::cout << "Order " << k << ":" << std::endl;
        std::cout << "  gamma[" << k << "] = " << gamma[k] << std::endl;
        std::cout << "  alpha[" << k << "] = " << alpha[k] << std::endl;
        std::cout << "  error_const[" << k << "] = " << error_const[k] << std::endl;
        std::cout << std::endl;
    }
    
    std::cout << "Expected for BDF1:" << std::endl;
    std::cout << "  gamma[1] = 1.0" << std::endl;
    std::cout << "  alpha[1] = 1.0" << std::endl;
    std::cout << "  error_const[1] = 0.5" << std::endl;
    
    return 0;
}
