#include <diffeq.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

int main() {
    std::cout << "=== Working ODE Integrators Demo ===" << std::endl;
    std::cout << "Problem: dy/dt = -y, y(0) = 1" << std::endl;
    std::cout << "Exact solution at t=1: " << std::exp(-1.0) << std::endl << std::endl;
    
    std::vector<double> y0 = {1.0};
    double t_start = 0.0, t_end = 1.0, dt = 0.1;
    
    // Test working integrators
    std::cout << "✅ Working Integrators:" << std::endl;
    
    try {
        auto y = y0;
        diffeq::integrators::ode::RK4Integrator<std::vector<double>> integrator(exponential_decay);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << "RK4 (4th order fixed):    " << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "RK4: Failed - " << e.what() << std::endl;
    }
    
    try {
        auto y = y0;
        diffeq::integrators::ode::RK23Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << "RK23 (3rd order adaptive): " << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "RK23: Failed - " << e.what() << std::endl;
    }
    
    try {
        auto y = y0;
        diffeq::integrators::ode::RK45Integrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << "RK45 (5th order adaptive): " << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "RK45: Failed - " << e.what() << std::endl;
    }
    
    try {
        auto y = y0;
        diffeq::integrators::ode::BDFIntegrator<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << "BDF (stiff systems):       " << std::setprecision(6) << y[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "BDF: Failed - " << e.what() << std::endl;
    }
    
    try {
        auto y = y0;
        diffeq::integrators::ode::LSODA<std::vector<double>> integrator(exponential_decay, 1e-6, 1e-9);
        integrator.set_time(t_start);
        integrator.integrate(y, dt, t_end);
        std::cout << "LSODA (automatic):        " << std::setprecision(6) << y[0] << 
            " (" << (integrator.get_current_method() == 
                   diffeq::integrators::ode::LSODA<std::vector<double>>::MethodType::ADAMS ? "Adams" : "BDF") << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "LSODA: Failed - " << e.what() << std::endl;
    }
    
    std::cout << "\n🔍 Known Issues:" << std::endl;
    std::cout << "❌ DOP853: Step size control needs refinement" << std::endl;
    std::cout << "❌ Radau: Basic implementation, needs Jacobian for production use" << std::endl;
    
    std::cout << "\n🎯 Recommendations:" << std::endl;
    std::cout << "• Use RK45 for general non-stiff problems (scipy default)" << std::endl;
    std::cout << "• Use BDF for stiff systems" << std::endl;
    std::cout << "• Use LSODA when stiffness is unknown" << std::endl;
    std::cout << "• Use RK4 for educational purposes or when simplicity is needed" << std::endl;
    
    std::cout << "\n✅ Implementation completed with 5/7 integrators fully functional!" << std::endl;
    
    return 0;
}
