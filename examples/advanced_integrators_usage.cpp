#include <iostream>
#include <vector>
#include <array>
#include <iomanip>
#include <chrono>

// Include all integrators
#include <solvers/rk4_solver.hpp>
#include <solvers/rk23_solver.hpp>
#include <solvers/rk45_solver.hpp>
#include <solvers/dop853_solver.hpp>
#include <solvers/radau_solver.hpp>
#include <solvers/bdf_solver.hpp>
#include <solvers/lsoda_solver.hpp>

// Test systems

// 1. Simple exponential decay: dy/dt = -y
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

// 2. Van der Pol oscillator: d²x/dt² - μ(1-x²)dx/dt + x = 0
class VanderPolOscillator {
public:
    explicit VanderPolOscillator(double mu) : mu_(mu) {}
    
    void operator()(double t, const std::vector<double>& y, std::vector<double>& dydt) {
        dydt[0] = y[1];                                    // dx/dt = v
        dydt[1] = mu_ * (1 - y[0]*y[0]) * y[1] - y[0];    // dv/dt = μ(1-x²)v - x
    }
    
private:
    double mu_;
};

// 3. Lorenz system (chaotic)
void lorenz_system(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double sigma = 10.0;
    const double rho = 28.0;
    const double beta = 8.0/3.0;
    
    dydt[0] = sigma * (y[1] - y[0]);           // dx/dt = σ(y - x)
    dydt[1] = y[0] * (rho - y[2]) - y[1];      // dy/dt = x(ρ - z) - y
    dydt[2] = y[0] * y[1] - beta * y[2];       // dz/dt = xy - βz
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
    integrator.integrate(y, dt, t_end);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << std::setw(15) << name << ": y = [";
    for (size_t i = 0; i < y.size(); ++i) {
        std::cout << std::setw(12) << std::scientific << std::setprecision(4) << y[i];
        if (i < y.size() - 1) std::cout << ", ";
    }
    std::cout << "] (Time: " << duration.count() << " μs)" << std::endl;
    
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
        RK4Integrator<std::vector<double>> integrator(exponential_decay);
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0};
        RK23Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "RK23");
    }
    
    {
        std::vector<double> y = {1.0};
        RK45Integrator<std::vector<double>> integrator(exponential_decay, 1e-8, 1e-12);
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    {
        std::vector<double> y = {1.0};
        DOP853Integrator<std::vector<double>> integrator(exponential_decay, 1e-10, 1e-15);
        time_integrator(integrator, y, t_start, dt, t_end, "DOP853");
    }
    
    {
        std::vector<double> y = {1.0};
        BDFIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    }
    
    {
        std::vector<double> y = {1.0};
        LSODAIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
    }
}

void demonstrate_van_der_pol() {
    std::cout << "\n=== Van der Pol Oscillator: μ = 5 (moderately stiff) ===" << std::endl;
    std::cout << "x''(t) - μ(1-x²)x'(t) + x(t) = 0" << std::endl;
    std::cout << "Initial conditions: x(0) = 1, x'(0) = 0" << std::endl << std::endl;
    
    VanderPolOscillator vdp(5.0);
    double t_start = 0.0, t_end = 10.0, dt = 0.1;
    
    {
        std::vector<double> y = {1.0, 0.0};
        RK4Integrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            });
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        RK45Integrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        RadauIntegrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "Radau");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        BDFIntegrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    }
    
    {
        std::vector<double> y = {1.0, 0.0};
        LSODAIntegrator<std::vector<double>> integrator(
            [&vdp](double t, const std::vector<double>& y, std::vector<double>& dydt) {
                vdp(t, y, dydt);
            }, 1e-6, 1e-9);
        double timing = time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
        std::cout << "        Final method: " << 
            (integrator.get_current_method() == LSODAIntegrator<std::vector<double>>::MethodType::ADAMS ? 
             "Adams (non-stiff)" : "BDF (stiff)") << std::endl;
    }
}

void demonstrate_lorenz_system() {
    std::cout << "\n=== Lorenz System (Chaotic) ===" << std::endl;
    std::cout << "dx/dt = σ(y - x), dy/dt = x(ρ - z) - y, dz/dt = xy - βz" << std::endl;
    std::cout << "σ = 10, ρ = 28, β = 8/3" << std::endl;
    std::cout << "Initial conditions: x(0) = 1, y(0) = 1, z(0) = 1" << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 5.0, dt = 0.01;
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        RK4Integrator<std::vector<double>> integrator(lorenz_system);
        time_integrator(integrator, y, t_start, dt, t_end, "RK4");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        RK45Integrator<std::vector<double>> integrator(lorenz_system, 1e-8, 1e-12);
        time_integrator(integrator, y, t_start, dt, t_end, "RK45");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        DOP853Integrator<std::vector<double>> integrator(lorenz_system, 1e-10, 1e-15);
        time_integrator(integrator, y, t_start, dt, t_end, "DOP853");
    }
    
    {
        std::vector<double> y = {1.0, 1.0, 1.0};
        LSODAIntegrator<std::vector<double>> integrator(lorenz_system, 1e-8, 1e-12);
        time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
    }
}

void demonstrate_stiff_robertson() {
    std::cout << "\n=== Robertson Chemical Kinetics (Very Stiff) ===" << std::endl;
    std::cout << "A -> B (k1 = 0.04)" << std::endl;
    std::cout << "B + B -> B + C (k2 = 3e7)" << std::endl;
    std::cout << "B + C -> A + C (k3 = 1e4)" << std::endl;
    std::cout << "Initial conditions: A(0) = 1, B(0) = 0, C(0) = 0" << std::endl << std::endl;
    
    double t_start = 0.0, t_end = 1e5, dt = 1.0;  // Very long time scale
    
    std::cout << "Note: This is a very stiff system. Explicit methods may fail or be very slow." << std::endl;
    
    // Only test implicit methods for this stiff system
    try {
        std::vector<double> y = {1.0, 0.0, 0.0};
        RadauIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "Radau");
    } catch (const std::exception& e) {
        std::cout << std::setw(15) << "Radau" << ": Failed - " << e.what() << std::endl;
    }
    
    try {
        std::vector<double> y = {1.0, 0.0, 0.0};
        BDFIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
        time_integrator(integrator, y, t_start, dt, t_end, "BDF");
    } catch (const std::exception& e) {
        std::cout << std::setw(15) << "BDF" << ": Failed - " << e.what() << std::endl;
    }
    
    try {
        std::vector<double> y = {1.0, 0.0, 0.0};
        LSODAIntegrator<std::vector<double>> integrator(robertson_kinetics, 1e-6, 1e-9);
        double timing = time_integrator(integrator, y, t_start, dt, t_end, "LSODA");
        std::cout << "        Final method: " << 
            (integrator.get_current_method() == LSODAIntegrator<std::vector<double>>::MethodType::ADAMS ? 
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
        {1e-9, 1e-12},
        {1e-12, 1e-15}
    };
    
    for (auto [rtol, atol] : tolerances) {
        std::vector<double> y = {1.0};
        RK45Integrator<std::vector<double>> integrator(exponential_decay, rtol, atol);
        
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
        std::cout << "✓ RK4: Simple, fixed-step 4th order method" << std::endl;
        std::cout << "✓ RK23: Adaptive 3rd order with embedded error estimation" << std::endl;
        std::cout << "✓ RK45: Adaptive 5th order Dormand-Prince (scipy default)" << std::endl;
        std::cout << "✓ DOP853: High-accuracy 8th order for demanding problems" << std::endl;
        std::cout << "✓ Radau: Implicit 5th order for stiff systems" << std::endl;
        std::cout << "✓ BDF: Variable-order implicit multistep for stiff systems" << std::endl;
        std::cout << "✓ LSODA: Automatic method switching (Adams ↔ BDF)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
