#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <core/concepts.hpp>
#include <core/abstract_integrator.hpp>
#include <core/state_creator.hpp>
#include <integrators/ode/rk4.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test helper macro
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAIL: " << message << " at line " << __LINE__ << std::endl; \
            return false; \
        } else { \
            std::cout << "PASS: " << message << std::endl; \
        } \
    } while(0)

// Simple ODE: dy/dt = -y (exponential decay)
// Analytical solution: y(t) = y0 * exp(-t)
void exponential_decay(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    dydt[0] = -y[0];
}

void exponential_decay_float(float t, const std::vector<float>& y, std::vector<float>& dydt) {
    dydt[0] = -y[0];
}

// Harmonic oscillator: d²x/dt² = -ω²x
// State vector: [x, dx/dt]
// dy/dt = [y[1], -ω²*y[0]]
void harmonic_oscillator(double t, const std::vector<double>& y, std::vector<double>& dydt) {
    const double omega_squared = 1.0; // ω² = 1
    dydt[0] = y[1];           // dx/dt = v
    dydt[1] = -omega_squared * y[0];  // dv/dt = -ω²x
}

bool test_rk4_double() {
    std::cout << "\n=== Testing RK4 with double precision ===" << std::endl;
    
    // Test exponential decay
    diffeq::RK4Integrator<std::vector<double>> integrator(exponential_decay);
    
    std::vector<double> state = {1.0}; // Initial condition: y(0) = 1
    double dt = 0.1;
    double end_time = 1.0;
    
    integrator.integrate(state, dt, end_time);
    
    // Analytical solution at t=1: y(1) = exp(-1) ≈ 0.3679
    double analytical = std::exp(-1.0);
    double error = std::abs(state[0] - analytical);
    
    std::cout << "Numerical solution: " << state[0] << std::endl;
    std::cout << "Analytical solution: " << analytical << std::endl;
    std::cout << "Error: " << error << std::endl;
    
    TEST_ASSERT(error < 1e-4, "RK4 double precision error should be small");
    
    return true;
}

bool test_rk4_float() {
    std::cout << "\n=== Testing RK4 with float precision ===" << std::endl;
    
    // Test exponential decay with float
    diffeq::RK4Integrator<std::vector<float>> integrator(exponential_decay_float);
    
    std::vector<float> state = {1.0f}; // Initial condition: y(0) = 1
    float dt = 0.1f;
    float end_time = 1.0f;
    
    integrator.integrate(state, dt, end_time);
    
    // Analytical solution at t=1: y(1) = exp(-1) ≈ 0.3679
    float analytical = std::exp(-1.0f);
    float error = std::abs(state[0] - analytical);
    
    std::cout << "Numerical solution: " << state[0] << std::endl;
    std::cout << "Analytical solution: " << analytical << std::endl;
    std::cout << "Error: " << error << std::endl;
    
    TEST_ASSERT(error < 1e-3f, "RK4 float precision error should be reasonable");
    
    return true;
}

bool test_rk4_harmonic_oscillator() {
    std::cout << "\n=== Testing RK4 with harmonic oscillator ===" << std::endl;
    
    diffeq::RK4Integrator<std::vector<double>> integrator(harmonic_oscillator);
    
    // Initial conditions: x(0) = 1, v(0) = 0
    std::vector<double> state = {1.0, 0.0};
    double dt = 0.01;
    double period = 2.0 * M_PI; // One complete oscillation
    
    integrator.integrate(state, dt, period);
    
    // After one period, should return close to initial conditions
    double x_error = std::abs(state[0] - 1.0);
    double v_error = std::abs(state[1] - 0.0);
    
    std::cout << "Final position: " << state[0] << " (should be ~1.0)" << std::endl;
    std::cout << "Final velocity: " << state[1] << " (should be ~0.0)" << std::endl;
    std::cout << "Position error: " << x_error << std::endl;
    std::cout << "Velocity error: " << v_error << std::endl;
    
    TEST_ASSERT(x_error < 0.01, "Harmonic oscillator position error should be small");
    TEST_ASSERT(v_error < 0.01, "Harmonic oscillator velocity error should be small");
    
    return true;
}

bool test_array_state() {
    std::cout << "\n=== Testing RK4 with std::array state ===" << std::endl;
    
    // Test with fixed-size array
    auto array_sys = [](double t, const std::array<double, 2>& y, std::array<double, 2>& dydt) {
        dydt[0] = y[1];
        dydt[1] = -y[0]; // Simple harmonic oscillator
    };
    
    diffeq::RK4Integrator<std::array<double, 2>> integrator(array_sys);
    
    std::array<double, 2> state = {1.0, 0.0};
    double dt = 0.01;
    double quarter_period = M_PI / 2.0; // Quarter oscillation
    
    integrator.integrate(state, dt, quarter_period);
    
    // After quarter period: x should be ~0, v should be ~-1
    double x_error = std::abs(state[0] - 0.0);
    double v_error = std::abs(state[1] - (-1.0));
    
    std::cout << "Final position: " << state[0] << " (should be ~0.0)" << std::endl;
    std::cout << "Final velocity: " << state[1] << " (should be ~-1.0)" << std::endl;
    
    TEST_ASSERT(x_error < 0.05, "Array state position error should be small");
    TEST_ASSERT(v_error < 0.05, "Array state velocity error should be small");
    
    return true;
}

int main() {
    std::cout << "Running RK4 Integrator Test Suite" << std::endl;
    std::cout << "==================================" << std::endl;
    
    bool all_passed = true;
    
    try {
        all_passed &= test_rk4_double();
        all_passed &= test_rk4_float();
        all_passed &= test_rk4_harmonic_oscillator();
        all_passed &= test_array_state();
        
        std::cout << "\n=== Test Results ===" << std::endl;
        if (all_passed) {
            std::cout << "🎉 All tests PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ Some tests FAILED!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Test execution failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test execution failed with unknown exception" << std::endl;
        return 1;
    }
}
