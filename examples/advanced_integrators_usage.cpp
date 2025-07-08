#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <chrono>

// Include modern diffeq library
#include <diffeq.hpp>

// Test systems

// 1. Simple exponential decay: dy/dt = -y
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

// 2. Van der Pol oscillator: d^2x/dt^2 - mu*(1-x^2)*dx/dt + x = 0
class VanderPolOscillator {
public:
    explicit VanderPolOscillator(double mu) : mu_(mu) {}
    
    void operator()(double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];                                    // dx/dt = v
        dydt[1] = mu_ * (1 - y[0]*y[0]) * y[1] - y[0];    // dv/dt = mu*(1-x^2)*v - x
    }
    
private:
    double mu_;
};

// 3. Lorenz system (chaotic)
void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0;
    const double rho = 28.0;
    const double beta = 8.0/3.0;
    
    dydt[0] = sigma * (y[1] - y[0]);           // dx/dt = sigma*(y - x)
    dydt[1] = y[0] * (rho - y[2]) - y[1];      // dy/dt = x*(rho - z) - y
    dydt[2] = y[0] * y[1] - beta * y[2];       // dz/dt = xy - beta*z
}

// 4. Stiff test problem: Robertson chemical kinetics
void robertson_kinetics(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double k1 = 0.04;
    const double k2 = 3e7;
    const double k3 = 1e4;
    
    dydt[0] = -k1 * y[0] + k3 * y[1] * y[2];                    // dA/dt
    dydt[1] = k1 * y[0] - k2 * y[1] * y[1] - k3 * y[1] * y[2];  // dB/dt
    dydt[2] = k2 * y[1] * y[1];                                 // dC/dt
}

// Helper function to time integrator performance
template<typename Integrator>
double time_integrator(Integrator& integrator, std::vector<double>& y, 
                      double t_start, double dt, double t_end, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    
    integrator.set_time(t_start);
    
    // Use timeout protection to prevent hanging
    const std::chrono::seconds TIMEOUT{30};  // 30 second timeout for more complex integrators
    bool completed = diffeq::integrate_with_timeout(integrator, y, dt, t_end, TIMEOUT);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << std::setw(15) << name << ": y = [";
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << std::setw(12) << std::scientific << std::setprecision(4) << y[i];
        if (i < y.size() - 1) std::cout << ", ";
    }
    std::cout << "] (Time: " << duration.count() << " us)";
    
    if (!completed) {
        std::cout << " [TIMEOUT]";
    }
    std::cout << std::endl;
    
    return static_cast<double>(duration.count());
}

void demonstrate_exponential_decay() {
    std::cout << "\n=== Exponential Decay: dy/dt = -y, y(0) = 1 ===" << std::endl;
    std::cout << "Analytical solution: y(t) = exp(-t)" << std::endl;
    std::cout << "At t = 1: y(1) = " << std::exp(-1.0) << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    
    // Test all integrators
    {
        std::vector<double> y = {1.0};
        diffeq::RK4Integrator<std::vector<double>> integrator(exponential_decay);
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0};
        diffeq::RK23Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "RK23");
    }
    
    {
        std::vector<double> y = {1.0};
        diffeq::RK45Integrator<std::vector<double>> integrator(exponential_decay, 1e-8, 1e-12);
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    {
        std::vector<double> y = {1.0};
        diffeq::DOP853Integrator<std::vector<double>> integrator(exponential_decay, 1e-3, 1e-6);
        time_integrator(integrator, y, t_start, dt, t_end, "DOP853");
    }
    
    {
        std::vector<double> y = {1.0};
        diffeq::BDFIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    }
    
    {
        std::vector<double> y = {1.0};
        diffeq::LSODAIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
    }
}

void demonstrate_van_der_pol() {
    std::cout << "\n=== Van der Pol Oscillator: mu = 5 (moderately stiff) ===" << std::endl;
    std::cout << "x''(t) - mu*(1-x^2)*x'(t) + x(t) = 0" << std::endl;
    std::cout << "Initial conditions: x(0) = 1, x'(0) = 0" << std::endl << std::endl;
    
    VanderPolOscillator vdp(5.0);
    double t_start = 0.0, t_end = 10.0, dt = 0.1;
    
    {
        std::vector<double> y = {1.0, 0.0};
        diffeq::RK4Integrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            });
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        diffeq::RK45Integrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    // Note: RadauIntegrator not implemented in current hierarchy
    // {
    //     std::vector<double> y = {1.0, 0.0};
    //     RadauIntegrator<std::vector<double>> integrator(
    //         [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
    //             vdp(t, y, dydt);
    //         }, 1e-6, 1e-9);
    //     time_integrator(integrator, y, t_start, dt, t_end, "Radau");
    // }
    
    {
        std::vector<double> y = {1.0, 0.0};
        diffeq::BDFIntegrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        diffeq::LSODAIntegrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        double timing = time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
        std::cout << "        Final method: " << 
            (integrator.get_current_method() == diffeq::LSODAIntegrator<std::vector<double>>::MethodType::ADAMS ? 
             "Adams (non-stiff)" : "BDF (stiff)") << std::endl;
    }
}

void demonstrate_lorenz_system() {
    std::cout << "\n=== Lorenz System (Chaotic) ===" << std::endl;
    std::cout << "dx/dt = sigma*(y - x), dy/dt = x*(rho - z) - y, dz/dt = xy - beta*z" << std::endl;
    std::cout << "sigma = 10, rho = 28, beta = 8/3" << std::endl;
    std::cout << "Initial conditions: x(0) = 1, y(0) = 1, z(0) = 1" << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 0.5, dt = 0.01;  // Reduced from 5.0 to 0.5 for faster execution
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        diffeq::RK4Integrator<std::vector<double>> integrator(lorenz_system);
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        diffeq::RK45Integrator<std::vector<double>> integrator(lorenz_system, 1e-6, 1e-9);  // Relaxed tolerances
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        diffeq::DOP853Integrator<std::vector<double>> integrator(lorenz_system, 1e-6, 1e-9);  // Relaxed tolerances
        time_integrator(integrator, y, t_start, dt, t_end, "DOP853");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        diffeq::LSODAIntegrator<std::vector<double>> integrator(lorenz_system, 1e-6, 1e-9);  // Relaxed tolerances
        time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
    }
}

void demonstrate_stiff_robertson() {
    std::cout << "\n=== Robertson Chemical Kinetics (Very Stiff) ===" << std::endl;
    std::cout << "A -> B (k1 = 0.04)" << std::endl;
    std::cout << "B + B -> B + C (k2 = 3e7)" << std::endl;
    std::cout << "B + C -> A + C (k3 = 1e4)" << std::endl;
    std::cout << "Initial conditions: A(0) = 1, B(0) = 0, C(0) = 0" << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1.0, dt = 0.1;  // Much shorter time scale for demo
    
    std::cout << "Note: This is a very stiff system. Explicit methods may fail or be very slow." << std::endl;
    std::cout << "Using shortened time range (t=1.0) for demonstration purposes." << std::endl;
    
    // Only test implicit methods for this stiff system
    // Note: RadauIntegrator not implemented in current hierarchy
    // try {
    //     std::vector<double> y = {1.0, 0.0, 0.0};
    //     RadauIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
    //     time_integrator(integrator, y, t_start, dt, t_end, "Radau");
    // } catch (const std::exception& e) {
    //     std::cout << std::setw(15) << "Radau" << ": Failed - " << e.what() << std::endl;
    // }
    
    try {
        std::vector<double> y = {1.0, 0.0, 0.0};
        diffeq::BDFIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    } catch (const std::exception& e) {
        std::cout << std::setw(15) << "BDF" << ": Failed - " << e.what() << std::endl;
    }
    
    try {
        std::vector<double> y = {1.0, 0.0, 0.0};
        diffeq::LSODAIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
        double timing = time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
        std::cout << "        Final method: " << 
            (integrator.get_current_method() == diffeq::LSODAIntegrator<std::vector<double>>::MethodType::ADAMS ? 
             "Adams (non-stiff)" : "BDF (stiff)") << std::endl;
    } catch (const std::exception& e) {
        std::cout << std::setw(15) << "LSODA" << ": Failed - " << e.what() << std::endl;
    }
}

void demonstrate_adaptive_features() {
    std::cout << "\n=== Adaptive Step Size Control Demonstration ===" << std::endl;
    std::cout << "Using RK45 with different tolerance levels on exponential decay" << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 2.0, dt = 0.1;
    
    std::vector<std::pair<double, double>> tolerances = {
        {1e-3, 1e-6},
        {1e-6, 1e-9}, 
        {1e-8, 1e-11}  // Removed extremely tight tolerances to prevent timeout
    };
    
    for (auto [rtol, atol] : tolerances) {
        std::vector<double> y = {1.0};
        diffeq::RK45Integrator<std::vector<double>> integrator(exponential_decay, rtol, atol);
        
        std::cout << "Tolerances: rtol = " << rtol << ", atol = " << atol << std::endl;
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
        
        double exact = std::exp(-t_end);
        double error = std::abs(y[0] - exact);
        std::cout << "        Error: " << std::scientific << error << std::endl << std::endl;
    }
}

int main() {
    std::cout << "Advanced ODE Integrators Demonstration" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        demonstrate_exponential_decay();
        demonstrate_van_der_pol();
        demonstrate_lorenz_system();
        demonstrate_stiff_robertson();
        demonstrate_adaptive_features();
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "[OK] RK4: Simple, fixed-step 4th order method" << std::endl;
        std::cout << "[OK] RK23: Adaptive 3rd order with embedded error estimation" << std::endl;
        std::cout << "[OK] RK45: Adaptive 5th order Dormand-Prince (scipy default)" << std::endl;
        std::cout << "[OK] DOP853: High-accuracy 8th order for demanding problems" << std::endl;
        std::cout << "- Radau: Implicit 5th order for stiff systems (not yet implemented)" << std::endl;
        std::cout << "[OK] BDF: Variable-order implicit multistep for stiff systems" << std::endl;
        std::cout << "[OK] LSODA: Automatic method switching (Adams <-> BDF)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
