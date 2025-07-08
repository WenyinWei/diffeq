#include <iostream>
#include <vector>
#include <cmath>

// Simple exponential decay system: dy/dt = -y
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

// Simple fixed-step BDF1 implementation
class SimpleBDF1 {
private:
    static constexpr int NEWTON_MAXITER = 4;
    static constexpr double NEWTON_TOL = 1e-10;
    
    double current_time_;
    
    double estimate_jacobian_diagonal(const std::vector<double>& y, double t) {
        // For exponential decay dy/dt = -y, we know J = -1
        // But let's estimate it numerically to match the real implementation
        double epsilon = 1e-8;
        std::vector<double> f_orig(1), f_pert(1);
        std::vector<double> y_pert = y;
        
        exponential_decay(t, y, f_orig);
        y_pert[0] += epsilon;
        exponential_decay(t, y_pert, f_pert);
        
        return (f_pert[0] - f_orig[0]) / epsilon;
    }
    
public:
    SimpleBDF1() : current_time_(0.0) {}
    
    void set_time(double t) { current_time_ = t; }
    double current_time() const { return current_time_; }
    
    bool step(std::vector<double>& y, double dt) {
        double t_new = current_time_ + dt;
        std::vector<double> y_new = y;
        
        // Initial guess: explicit Euler
        std::vector<double> f_current(1);
        exponential_decay(current_time_, y, f_current);
        y_new[0] = y[0] + dt * f_current[0];
        
        std::cout << "  Initial guess: " << y_new[0] << std::endl;
        
        // Newton iteration
        for (int iter = 0; iter < NEWTON_MAXITER; ++iter) {
            // Evaluate function at current guess
            std::vector<double> f(1);
            exponential_decay(t_new, y_new, f);
            
            // Compute residual: F(y_new) = y_new - dt * f(t_new, y_new) - y_current
            double residual = y_new[0] - dt * f[0] - y[0];
            
            std::cout << "  Iter " << iter << ": y=" << y_new[0] << ", f=" << f[0] 
                      << ", residual=" << residual << std::endl;
            
            // Check convergence
            if (std::abs(residual) < NEWTON_TOL) {
                std::cout << "  Converged after " << iter+1 << " iterations" << std::endl;
                y = y_new;
                current_time_ = t_new;
                return true;
            }
            
            // Newton step: solve (1 - dt*J) * dy = -residual
            double jac_diag = estimate_jacobian_diagonal(y_new, t_new);
            double denominator = 1.0 - dt * jac_diag;
            
            std::cout << "  jac=" << jac_diag << ", denom=" << denominator << std::endl;
            
            if (std::abs(denominator) > 1e-12) {
                double dy = -residual / denominator;
                y_new[0] += dy;
                std::cout << "  dy=" << dy << ", new_y=" << y_new[0] << std::endl;
            } else {
                std::cout << "  Singular Jacobian!" << std::endl;
                return false;
            }
        }
        
        std::cout << "  Newton failed to converge" << std::endl;
        return false;
    }
};

int main() {
    std::cout << "=== Simple Fixed BDF1 Debug ===" << std::endl;
    
    SimpleBDF1 integrator;
    std::vector<double> y = {1.0};
    double dt = 0.01;
    
    std::cout << "Testing multiple steps with dt=" << dt << std::endl;
    
    for (int i = 0; i < 5; ++i) {
        double y_before = y[0];
        double t_before = integrator.current_time();
        
        std::cout << "\nStep " << i << ": t=" << t_before << ", y=" << y_before << std::endl;
        
        bool success = integrator.step(y, dt);
        
        double y_after = y[0];
        double t_after = integrator.current_time();
        double expected = y_before / (1.0 + dt);
        double error = std::abs(y_after - expected);
        
        std::cout << "Result: t=" << t_after << ", y=" << y_after 
                  << ", expected=" << expected << ", error=" << error << std::endl;
        
        if (!success) {
            std::cout << "Step failed!" << std::endl;
            break;
        }
        
        if (error > 1e-10) {
            std::cout << "ERROR: Significant error in step " << i << std::endl;
            break;
        }
    }
    
    return 0;
}
